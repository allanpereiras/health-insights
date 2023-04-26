"""
Determines the URL conf for Insights API using routers to automatically
register all the endpoints with the corresponding Viewset.

For example, you might have a `urls.py` that looks something like this:
    router = routers.DefaultRouter()
    router.register('users', UserViewSet, 'user')
    router.register('accounts', AccountViewSet, 'account')

    urlpatterns = router.urls
"""

from django.urls import include, path
from rest_framework import routers

from . import viewsets

app_name = "insights"


class InsightsApiView(routers.APIRootView):
    """Insights App API Root View"""

    pass


class InsightsRouter(routers.DefaultRouter):
    """Insights Router"""

    APIRootView = InsightsApiView


router = InsightsRouter()
router.register(r"data-distributor-ledgers", viewsets.DataDistributorLedgerViewSet)
router.register(r"health-insights", viewsets.HealthInsightViewSet)
router.register(r"markers", viewsets.MarkerViewSet)
router.register(r"patients-health-insights", viewsets.PatientHealthInsightViewSet)
router.register(
    r"patients-recommendations",
    viewsets.PatientRecommendationsViewSet,
    basename="patients-recommendations",
)

urlpatterns = [
    path("api/insights/", include(router.urls)),
]
