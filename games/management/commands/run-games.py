import pickle
import os.path
from datetime import datetime
from api import State, engine, util

from bots.ml import ml
from bots.ml.ml import features as ml_bot_features

from bots.ml_missing_feature import ml_missing_feature
from bots.ml_missing_feature.ml_missing_feature import features as mf_bot_features

import sklearn
import sklearn.linear_model
from sklearn.neural_network import MLPClassifier
import joblib

from django.core.management.base import BaseCommand
from games.models import GameResult

GAMES = 100

TRAINING_GAMES = 10000

STATE = State.generate()

def bot_name(bot):
    return bot.__module__.split('.')[2]

def report(player1, player2, winner, points, inspected, trained_against, excluded_feature=None,):
    player1_name = bot_name(player1)
    player2_name = bot_name(player2)

    if winner == 1:
        winner_name = player1_name
    elif winner == 2:
        winner_name = player2_name

    GameResult.objects.create(
        player1=player1_name,
        player2=player2_name,
        winner=winner_name,
        winner_points=points,
        excluded_feature=excluded_feature,
        inspected=inspected,
        trained_against=trained_against
    )


def play(player1, player2, inspected, trained_against, games=GAMES, excluded_feature=None):
    for i in range(games//2):
        state = State.generate()

        winner, points = engine.play(player1, player2, state=state, verbose=False)

        report(player1, player2, winner, points, inspected, trained_against, excluded_feature)

        winner, points = engine.play(player2, player1, state=state, verbose=False)

        report(player2, player1, winner, points, inspected, trained_against, excluded_feature)

def create_dataset(path, player, feature_func, included_features, games=TRAINING_GAMES):
    data = []
    target = []

    for g in range(games-1):
        state = State.generate()
        state_vectors = []

        while not state.finished():
            given_state = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state
            state_vectors.append(feature_func(given_state, included_features))
            move = player.get_move(given_state)
            state = state.next(move)

        winner, score = state.winner()

        for state_vector in state_vectors:
            data.append(state_vector)

            if winner == 1:
                result = 'won'
            elif winner == 2:
                result = 'lost'

            target.append(result)

    with open(path, 'wb') as output:
        pickle.dump((data, target), output, pickle.HIGHEST_PROTOCOL)

    return data, target

def train_bot(dataset_file, model_file):
    # Train the machine learning bots
    hidden_layer_sizes = (64, 32)
    learning_rate = 0.0001
    regularization_strength = 0.0001

    with open(dataset_file, 'rb') as output:
        data, target = pickle.load(output)

    learner = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate, alpha=regularization_strength, verbose=False, early_stopping=True, n_iter_no_change=6)

    model = learner.fit(data, target)

    count = {}
    for t in target:
        if t not in count:
            count[t] = 0
        count[t] += 1

    joblib.dump(model, model_file)


class Command(BaseCommand):
    help = ''

    def handle(self, *args, **options):
        all_features = [
            'perspective',
            'p1_points',
            'p2_points',
            'p1_pending_points',
            'p2_pending_points',
            'trump_suit_onehot',
            'phase',
            'stock_size',
            'whose_turn',
            'opponents_played_card_onehot',
        ]

        opponents = [
            util.load_player('rand'),
            util.load_player('bully'),
            util.load_player('rdeep'),
        ]

        for opponent in opponents:
            print('Start experiment against', bot_name(opponent))
            # Generate the datasets
            ml_model_file = f'./bots/ml/{bot_name(opponent)}-model.pkl'
            ml_dataset_file = f'datasets/ml-{bot_name(opponent)}-dataset.pkl'
            # create_dataset(ml_dataset_file, opponent, ml_bot_features, [])
            print('Created dataset', ml_dataset_file)
            # train_bot(ml_dataset_file, ml_model_file)
            print('Output training results to', ml_model_file)
            ml_bot = ml.Bot(model_file=ml_model_file)

            play(ml_bot, opponent, bot_name(ml_bot), bot_name(opponent))

            for excluded_feature in all_features:
                mf_model_file = f'./bots/ml_missing_feature/{bot_name(opponent)}-{excluded_feature}-model.pkl'
                mf_dataset_file = f'datasets/mf-{bot_name(opponent)}-{excluded_feature}-dataset.pkl'

                included_features = all_features.copy()
                included_features.remove(excluded_feature)

                # create_dataset(mf_dataset_file, opponent, mf_bot_features, included_features)
                print('Created dataset', mf_dataset_file)

                # train_bot(mf_dataset_file, mf_model_file)
                print('Output training results to', mf_model_file)

                missing_feature_bot = ml_missing_feature.Bot(mf_model_file, included_features)

                play(missing_feature_bot, opponent, bot_name(missing_feature_bot), bot_name(opponent), excluded_feature=excluded_feature)

                play(missing_feature_bot, ml_bot, bot_name(missing_feature_bot), bot_name(opponent), excluded_feature=excluded_feature)
                print('Finished tournaments against', bot_name(opponent), 'with missing feature', excluded_feature)
