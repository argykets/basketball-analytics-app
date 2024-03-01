import numpy as np
import typer
from config import SAVED_MODEL_LOC
from typing_extensions import Annotated
from utils import load_model

app = typer.Typer()


@app.command()
def predict(
    shot_clock: Annotated[float, typer.Option()],
    dribbles: Annotated[int, typer.Option()],
    touch_time: Annotated[float, typer.Option()],
    shot_dist: Annotated[float, typer.Option()],
    pts_type: Annotated[int, typer.Option()],
    close_def_dist: Annotated[float, typer.Option()],
) -> float:
    """Uses pretrained model for predicting new instances

    Args:
        shot_clock (Annotated[float, typer.Option): Shot clock in seconds
        dribbles (Annotated[int, typer.Option): Number of dribbles before shooting
        touch_time (Annotated[float, typer.Option): Touch time before shooting in seconds
        shot_dist (Annotated[float, typer.Option): Shot distances in meters
        pts_type (Annotated[int, typer.Option): 2 if the shot is 2-pt and 3 if it is 3-pt
        close_def_dist (Annotated[float, typer.Option): Distance of closest defender in meters

    Returns:
        float: 0.0 if missing shot, 1.0 if made shot
    """
    # Load saved model
    model = load_model(SAVED_MODEL_LOC)

    input_data = np.array([shot_clock, dribbles, touch_time, shot_dist, pts_type, close_def_dist])
    input_data = input_data.reshape(1, len(input_data))

    prediction = model.predict(input_data)
    return prediction


if __name__ == "__main__":
    app()
