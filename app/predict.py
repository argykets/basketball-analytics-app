import typer
import numpy as np
from utils import load_model
from config import SAVED_MODEL_LOC
from typing_extensions import Annotated

app = typer.Typer()

@app.command()
def predict(
    shot_clock: Annotated[float, typer.Option()],
    dribbles: Annotated[int, typer.Option()],
    touch_time: Annotated[float, typer.Option()],
    shot_dist: Annotated[float, typer.Option()],
    pts_type: Annotated[int, typer.Option()],
    close_def_dist: Annotated[float, typer.Option()]
):
    # Load saved model
    model = load_model(SAVED_MODEL_LOC)

    input_data = np.array([shot_clock, dribbles, touch_time,
                            shot_dist, pts_type, close_def_dist])
    input_data = input_data.reshape(1, len(input_data))

    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    app()