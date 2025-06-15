import dill

import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()