"""
ViewSets are a type of class based view, that doesn't provide any method
handlers, such as `get()`, `post()`, etc. but instead has actions,
such as `list()`, `retrieve()`, `create()`, etc.

Actions are only bound to methods at the point of instantiating the views.

.. code-block:: python

    user_list = UserViewSet.as_view({'get': 'list'})
    user_detail = UserViewSet.as_view({'get': 'retrieve'})

Typically, rather than instantiate views from viewsets directly, you'll
register the viewset with a router and let the URL conf be determined
automatically.

.. code-block:: python

    router = DefaultRouter()
    router.register(r'users', UserViewSet, 'user')
    urlpatterns = router.urls
"""
from datetime import datetime

import pytz
from django.conf import settings
from django.db import OperationalError
from django.utils.timezone import localdate, localtime, make_aware, utc
from django_filters import rest_framework as drf_filter
from insights import models
from patients.utils import get_or_create_patient_with_external_id as get_patient
from rest_framework import exceptions, status, viewsets
from rest_framework.decorators import action
from rest_framework.mixins import UpdateModelMixin
from rest_framework.response import Response

from . import filters, serializers
from .tasks import get_or_create_all_insights_for_patient
from .utils import RedshiftUnavailable


class DataDistributorLedgerViewSet(viewsets.ReadOnlyModelViewSet, UpdateModelMixin):
    """
    Provides default `list()`, `retrieve()`, `update()`, `partial_update()`
    actions for DataDistributorLedger model.
    `get_ledger_batch()` returns a list of 10 ledger entries.
    """

    queryset = models.DataDistributorLedger.objects.all().order_by("-started_at")
    serializer_class = serializers.DataDistributorLedgerSerializer
    filter_backends = (drf_filter.DjangoFilterBackend,)
    filterset_class = filters.DataDistributorLedgerFilter
    lookup_field = "uuid"

    REGEX_URL = {
        "get_ledger_batch": r"batch/(?P<ext_patient_id>\w+)/(?P<distrib_name>[\w|\s]+)",
    }

    @action(
        detail=False,
        methods=["get"],
        name="Ledger Batch",
        url_path=REGEX_URL["get_ledger_batch"],
    )
    def get_ledger_batch(self, request, ext_patient_id, distrib_name):
        """
        Returns a list of 12 Data Distributor Ledger entries.

        :param ext_patient_id:  External Patient ID.
        :type ext_patient_id:   str
        :param distrib_name:    Data Distributor name.
        :type distrib_name:     str
        :raise Http404:         If Patient or DataDistributor were not found.
        :return:                The Data Distributor Ledgers list.
        :rtype:                 list[str]
        """
        try:
            patient, _ = get_patient(ext_patient_id)
            kwargs = {
                "patient": patient,
                "distributor": models.DataDistributor.objects.get(
                    name__iexact=distrib_name
                ),
                "batch_size": int(settings.LEDGERS_BATCH_SIZE),
            }
        except Exception as e:
            raise exceptions.NotFound from e
        batch = models.DataDistributorLedger.get_ledger_batch(**kwargs)

        page = self.paginate_queryset(batch)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(batch, many=True)
        return Response(serializer.data)


class MarkerViewSet(viewsets.ReadOnlyModelViewSet):
    """Provides default `list()`, `retrieve()` actions for Marker model."""

    queryset = models.Marker.objects.all()
    serializer_class = serializers.MarkerSerializer


class HealthInsightViewSet(viewsets.ReadOnlyModelViewSet):
    """Provides default `list`, `retrieve` actions for HealthInsight model."""

    queryset = models.HealthInsight.objects.all()
    serializer_class = serializers.HealthInsightSerializer


class PatientHealthInsightViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Provides default `list`, `retrieve`, `redshift_values`,
    `display`, `latest` actions for PatientHealthInsight model.
    For result filtering see :class:`~filters.PatientHealthInsightFilter`
    """

    queryset = models.PatientHealthInsight.objects.all().order_by("-collection_date")
    serializer_class = serializers.PatientInsightsDisplaySerializer
    filter_backends = (drf_filter.DjangoFilterBackend,)
    filterset_class = filters.PatientHealthInsightFilter

    @action(
        detail=False,
        methods=["get"],
        name="Redshift Patient's Distributor Insights",
        url_path=r"redshift/(?P<ext_patient_id>\w+)/(?P<distrib>[\w|\s]+)/(?P<collection_date>\d{4}-\d{2}-\d{2})",
    )
    def redshift_values(self, request, ext_patient_id, distrib, collection_date):
        """
        Returns a list of all calculated Health Insights for a Patient, Distributor and
        collection date.

        :param ext_patient_id:  External Patient ID.
        :type ext_patient_id:   str
        :param distrib_name:    Data Distributor name.
        :type distrib_name:     str
        :param collection_date: Collection date.
        :type collection_date:  str
        """
        tz = self.request.GET.get("timezone", str(utc))  # Timezone with default to UTC
        try:
            collection_date = datetime.strptime(collection_date, "%Y-%m-%d")
            utc_collection_dt = make_aware(
                collection_date, timezone=pytz.timezone(tz)
            ).astimezone(
                tz=utc
            )  # Validates Future date in UTC
            if utc_collection_dt > localtime():
                raise exceptions.ValidationError(
                    {"collection_date": "Collection date cannot be a future date"}
                )
            collection_date = str(collection_date.date())  # datetime to date
        except ValueError as e:
            raise exceptions.ParseError(
                f"Invalid collection date `{collection_date}`"
            ) from e
        if not models.DataDistributor.objects.filter(name__iexact=distrib).exists():
            raise exceptions.NotFound(f"Data Distributor `{distrib}` not found")
        kwargs = request.GET.dict()
        get_or_create_all_insights_for_patient.delay(
            ext_patient_id, distrib, collection_date, tz, **kwargs
        )
        args = {
            "ext_patient_id": ext_patient_id,
            "distrib": distrib,
            "collection_date": collection_date,
            "kwargs": kwargs,
        }
        return Response(f"Fired task to calculate insights for {args}")

    @action(
        detail=False,
        methods=["get"],
        name="Patient Insights for display",
        url_path=r"(?P<ext_patient_id>\w+)/display",
    )
    def display(self, request, ext_patient_id):
        """Returns list of Patient Insights for display according to
        HealthInsightDisplayRuleSet evaluation
        """
        try:
            patient, _ = get_patient(ext_patient_id)
        except:
            raise exceptions.NotFound(f"Patient `{ext_patient_id}` not found")
        all_rulesets = models.HealthInsightDisplayRuleSet.objects.all().order_by(
            "health_insight", "priority"
        )
        results = []
        found_insights = []
        try:
            for insight_ruleset in all_rulesets:
                patient_insight = insight_ruleset.get_for_patient_display(patient.id)
                if patient_insight:
                    results.append(patient_insight)
                    found_insights.append(patient_insight.health_insight)
                # Remove rulesets whose Health Insight have already been evaluated
                all_rulesets = all_rulesets.exclude(health_insight__in=found_insights)
        except (ValueError, RuntimeError) as value_error:
            # Failed to update ORM / possible constraints violation
            raise exceptions.ValidationError from value_error
        except OperationalError as e:
            # Failed querying Redshift
            raise RedshiftUnavailable from e
        page = self.paginate_queryset(results)
        srlz = serializers.PatientInsightsDisplaySerializer(results, many=True)
        paginated_response = self.get_paginated_response(srlz.data)
        return paginated_response if page is not None else Response(srlz.data)

    @action(
        detail=False,
        methods=["get"],
        name="Latest Patient Insights",
        url_path=r"(?P<ext_patient_id>\w+)/latest",
    )
    def latest(self, request, ext_patient_id):
        try:
            patient, _ = get_patient(ext_patient_id)
        except:
            raise exceptions.NotFound(f"Patient `{ext_patient_id}` not found")
        patient_insights = (
            models.PatientHealthInsight.objects.filter(patient=patient)
            .order_by("health_insight", "-collection_date")
            .distinct("health_insight")
        )
        page = self.paginate_queryset(patient_insights)
        srlz = serializers.PatientInsightsDisplaySerializer(patient_insights, many=True)
        paginated_response = self.get_paginated_response(srlz.data)
        return paginated_response if page is not None else Response(srlz.data)

    @action(
        detail=False,
        methods=["get"],
        name="Returns the average value of a Patient Health Insight for a period",
        url_path=r"chart/(?P<ext_patient_id>\w+)/(?P<h_insight_id>\d+)",
    )
    def chart(self, request, ext_patient_id, h_insight_id):
        """Returns the average value of a Patient Health Insight for a period
        Parameters
        ----------
        ext_patient_id : str (required) External Patient ID
        h_insight_id : int (required) Health Insight ID
        period : str (optional) default 'day' Aggregation period type
        to_date : str (optional) default today. End date of the aggregation period
        timezone : str (optional) default UTC. Timezone for collection_date period
        """
        try:
            health_insight = models.HealthInsight.objects.get(id=h_insight_id)
            tz = request.GET.get("timezone", str(utc))  # Timezone with default to UTC
            tz = pytz.timezone(tz)
            to_date = request.GET.get("to_date", str(localdate()))  # default to today
            to_date = datetime.strptime(to_date, "%Y-%m-%d")
            to_date = to_date.replace(tzinfo=tz)
        except models.HealthInsight.DoesNotExist as h:
            msg = f"Health Insight `{h_insight_id}` not found"
            raise exceptions.NotFound(msg) from h
        except pytz.UnknownTimeZoneError as t:
            raise exceptions.ValidationError(f"Invalid Timezone `{tz}`: {t}") from t
        except ValueError as d:  # datetime.strptime
            raise exceptions.ValidationError(f"Erro parsing arg `to_date`: {d}") from d

        period = request.GET.get("period", "day")  # default period is day
        if period not in serializers.HealthDataChartSerializer.PERIODS:
            raise exceptions.ValidationError(f"Invalid aggregation period `{period}`")
        start_date, end_date = serializers.HealthDataChartSerializer.get_period_dates(
            period, to_date
        )
        dict_obj = {
            "external_patient_id": ext_patient_id,
            "health_insight": health_insight,
            "period": period,
            "from_date": start_date,
            "to_date": end_date,
        }
        srlz = serializers.HealthDataChartSerializer(
            data=dict_obj, context={"request": request}
        )
        if not srlz.is_valid():
            return Response(srlz.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response(srlz.data)


class PatientRecommendationsViewSet(viewsets.ViewSet):
    """
    Provides `get_recommendations_for_patient` action to list Recommendations
    generated from Health Insights of a Patient
    """

    serializer_class = serializers.PatientHealthInsightRecommendationSerializer
    filter_backends = (drf_filter.DjangoFilterBackend,)
    filterset_class = filters.PatientRecommendationsFilter

    @action(
        detail=False,
        methods=["get"],
        name="Get list of recommendations for a Patient",
        url_path=r"(?P<ext_patient_id>\w+)",
    )
    def get_recommendations_for_patient(self, request, ext_patient_id):
        """Returns a list of recommendations for an external patient id"""
        active_only = self.request.query_params.get("active_only", "true")
        active_only = active_only.lower() == "true"
        dict_obj = {"external_patient_id": ext_patient_id}
        serializer = serializers.PatientHealthInsightRecommendationSerializer(
            dict_obj, many=False, active_only=active_only
        )
        return Response(serializer.data)
