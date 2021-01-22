from django.views.generic import UpdateView, CreateView, DeleteView, TemplateView

class DashboardView(TemplateView):
    template_name = 'dashboard.html'
