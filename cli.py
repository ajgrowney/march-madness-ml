import click
import os
import numpy as np
import json
import pickle
import uuid
from sklearn.preprocessing import StandardScaler
from utilities import evaluate_model_on_tournament, fetch_training_data, MODELS_ROOT
from sklearn.model_selection import train_test_split, GridSearchCV
CS_ALL_SEASONS = ",".join([str(i) for i in (range(2003,2019))])

@click.group()
def cli():
    pass


@click.command()
@click.option('--models', default="basic_svc")
@click.option('--save', is_flag=True)
@click.option('--scale', is_flag=True, default = True)
def train(models, save, scale):
    """Train a set of models by their architecture identifier
    """
    X,Y,features = fetch_training_data()
    X, Y = np.array(X), np.array(Y).reshape(-1,1)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.85)
    model_ids = models.split(",")
    trained_models = []
    metadata = {}

    for m in model_ids:
        if m == "basic_svc":
            from model_trainer import train_basic_svc
            model, score = train_basic_svc(x_train, x_test, y_train, y_test )
        elif m == "basic_xgb":
            from model_trainer import train_xgb_basic
            model, score = train_xgb_basic(x_train, x_test, y_train, y_test)
        elif m == "grid_xgb":
            from model_trainer import train_xgb_grid
            model, score, selected_params = train_xgb_grid(x_train, x_test, y_train, y_test)
            print(f"Grid XGB Params: {selected_params}")
        
        print(f"Model\t\tScore\n{m}\t{score}")
        trained_models.append((m, model))

    
    if save:
        run_id = uuid.uuid4()
        print(f"Saving models in run: {run_id}")
        run_root = f"{MODELS_ROOT}/{run_id}"
        os.makedirs(run_root)
        for m_id, model in trained_models:
            model_file_path = os.path.join(run_root, f"{m_id}.sav")
            pickle.dump(model, open(model_file_path, 'wb'))

        if scale:
            pickle.dump(scaler, open(os.path.join(run_root, "scaler.pkl"), "wb"))

    return

@click.command()
@click.option('--run', help="Run identifier")
@click.option('--models', default="basic_svc", help="Comma separated list of models to use")
@click.option('--seasons', default=CS_ALL_SEASONS, help="Comma separated list of seasons")
def evaluate(run, models, seasons):
    """Evaluate model on a historical tournament
    """
    models = models.split(",")
    seasons = [int(s) for s in seasons.split(",")]
    run_root = os.path.join(MODELS_ROOT, run)
    scaler_path = os.path.join(run_root, "scaler.pkl")
    
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, "rb"))
    else:
        scaler = None
    
    results = {}
    # Load and Evaluate models
    for m in models:
        model_path = os.path.join(run_root, f"{m}.sav")
        model = pickle.load(open(model_path,"rb"))
        for s in seasons:
            results[s] = evaluate_model_on_tournament(model, scaler, s)
    
    # Display Results
    [print(f"{k} - {v['score']}") for k,v in results.items()]
    return

# Fill Kaggle Submission with Model
@click.command()
def submit():
    return

cli.add_command(train)
cli.add_command(evaluate)
if __name__ == "__main__":
    cli()