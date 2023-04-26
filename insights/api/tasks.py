import logging
import traceback
from datetime import timedelta

from celery.utils.log import get_task_logger
from core.celery import app
from django.db.models import Q
from django.utils import timezone
from insights import models, redshift

celery_logger = get_task_logger(__name__)
logger = logging.getLogger("tasks")


@app.task
def get_or_create_all_insights_for_patient(ext_id, distrib, collec_date, tz, **kwargs):
    """Runs get_or_create_all_insights_for_patient in redshift.PatientInsights
    to trigger the creation of all insights for a Distributor's patient.
    Logs the number of insights returned for the patient or any errors.
    :param ext_pat_id: str - external patient id
    :param distrib: str - distributor name
    :param collec_date: str - collection date
    :param tz: str - timezone
    :param kwargs: dict - additional arguments forwards to redshift.PatientInsights
    :return: None
    """
    msg = "Task get_or_create_all_insights_for_patient just ran."
    celery_logger.info(msg)
    logger.info(msg)
    try:
        patient_insights = redshift.PatientInsights(ext_id, collec_date, tz)
        patient_insights.get_or_create_all_insights(distrib, **kwargs)
        rows = len(patient_insights.health_insights)
        msg = f"{rows} insights returned for patient {ext_id} in {collec_date}."
        celery_logger.info(msg)
        logger.info(msg)
    except (ValueError, RuntimeError) as value_error:
        msg = f"Failed to update ORM / possible constraints violation: {value_error}"
        celery_logger.error(msg)
        logger.error(msg)
    except Exception as e:
        msg = f"Celery task `get_or_create_all_insights_for_patient` failed: {e}"
        trail = str(traceback.format_exc())
        celery_logger.info(msg)
        celery_logger.info(trail)
        logger.info(msg)
        logger.info(trail)


@app.task
def process_insights_recommendations():
    """Identify patients with insights modified since yesterday
    and process their recommendations updates or creation.
    """
    task = "process_insights_recommendations"
    celery_logger.info(f"Task {task} just ran.")
    past_day = timezone.now() - timedelta(days=1)
    patients_to_process = models.PatientHealthInsight.objects.filter(
        Q(created_at__gte=past_day) | Q(updated_at__gte=past_day)
    ).values_list("patient_id", flat=True)
    for patient_id in patients_to_process.distinct():
        try:
            models.RecommendationTrigger.process_all.delay(patient_id)
        except Exception as e:
            msg = f"Celery task `{task}` failed for patient_id {patient_id}: {e}"
            trail = str(traceback.format_exc())
            celery_logger.info(msg)
            celery_logger.info(trail)
