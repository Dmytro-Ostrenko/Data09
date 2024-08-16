from django.shortcuts import render
from utils.py_logger import get_logger

logger = get_logger(__name__)


def index(request):
    logger.info("Started index")
    return render(request, 'index.html')

