from django.views.generic import TemplateView
from games.models import GameResult

class DashboardView(TemplateView):
    template_name = 'dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        q = GameResult.objects.all()[16000:17000]

        wins = 0
        loss = 0

        for result in q:
            if result.winner == 'ml_missing_feature':
                wins += 1
            else:
                loss += 1

        results = {
            'wins': wins,
            'loss': loss,
            'missing': q[0].excluded_feature,
            'p1': q[0].player1,
            'p2': q[0].player2,
        }


        context["results"] = results

        return context
