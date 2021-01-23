from django.db import models
from django.db.models import Q


class GameResult(models.Model):
    player1 = models.CharField(max_length=255, null=False)
    player2 = models.CharField(max_length=255, null=False)
    winner = models.CharField(max_length=255, null=False)
    winner_points = models.PositiveIntegerField(null=False)
    excluded_feature = models.CharField(max_length=255, null=True, default=None)
    created_at = models.DateTimeField(auto_now_add=True)
    inspected = models.CharField(max_length=255, null=False)
    trained_against = models.CharField(max_length=255, null=False)
