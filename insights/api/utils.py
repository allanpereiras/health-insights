from rest_framework.exceptions import APIException


class RedshiftUnavailable(APIException):
    """API Exception for 503 - Service Unavailable when querying Redshift"""
    status_code = 503
    default_detail = 'Redshift connection temporarily unavailable, try again later.'
    default_code = 'service_unavailable'
