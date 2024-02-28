from fastapi import FastAPI
from http import HTTPStatus
import uvicorn
from pydantic import BaseModel
import predict
from data import convert_data_to_features

app = FastAPI(
    title="Shot Predictor",
    description="Classify shots as made or missed",
    version="0.1"
)

class ShotData(BaseModel):
    shot_clock: float
    dribbles: int
    touch_time: float
    shot_dist: float
    pts_type: int
    close_def_dist: float

@app.get("/")
def _index():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response

@app.post("/predict/")
async def _predict(shot_data: ShotData):
    shot_clock, dribbles, touch_time, shot_dist, pts_type, close_def_dist = convert_data_to_features(shot_data)
    results = predict.predict(shot_clock=shot_clock, dribbles=dribbles, touch_time=touch_time, shot_dist=shot_dist,
                              pts_type=pts_type, close_def_dist=close_def_dist)
    
    return {"results": int(results[0])}


if __name__ == "__main__":
    uvicorn.run("serve:app", reload=True)