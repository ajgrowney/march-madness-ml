from email.policy import default
import click
import os
import numpy as np
import json
import pickle
import uuid
import shutil
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from march_madness_analytics.utilities import DATA_ROOT, evaluate_model_on_tournament, f_importances, fetch_training_data, MODELS_ROOT, fill_submission, get_scaler, SUBMISSIONS_ROOT
from bracketeer import build_bracket

CS_ALL_SEASONS = ",".join([str(i) for i in list(range(2003,2020)) + [2021]])

@click.group()
def cli():
    pass


@click.command()
@click.option('--models', default="basic_svc")
@click.option('--save', is_flag=True)
@click.option('--scale', default = None)
@click.option('--data', default = "ComputedV1")
@click.option('--data-format', default="diff")
@click.option('--regular-season', is_flag=True, default = False)
def train(models, save, scale, data, data_format, regular_season):
    """Train a set of models by their architecture identifier
    """
    train_size = 0.90
    metadata = {"Data": {"TrainingSource": data, "Scale": scale, "TrainSplit": train_size}} 
    
    # --- Data Retrieval ----
    metadata["TrainingData"] = data
    X,Y,features,tourney_idx = fetch_training_data(version=data, format=data_format, inclued_reg_season=regular_season)
    X, Y = np.array(X), np.array(Y).reshape(-1,1)
    
    
    if scale is not None:
        if scale == "MinMax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X = scaler.fit_transform(X)
    print(f"Train Size: {X.shape}")
    metadata["Data"]["TrainingSize"] = X.shape

    if regular_season:
        x_rs_train, y_rs_train = X[0:tourney_idx], Y[0:tourney_idx]
        X,Y = X[tourney_idx:], Y[tourney_idx:]
    
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.8, shuffle=True, stratify=Y)

    if regular_season:
        tourney_game_count = len(x_train)
        x_train = np.append(x_train, x_rs_train, axis=0)
        y_train = np.append(y_train, y_rs_train, axis=0)
        sample_weights = np.ones(len(x_train))
        for i in range(0, tourney_game_count):
            sample_weights[i] = 1
    else:
        sample_weights = None    

    # ---- Training ----
    trained_models = []

    for m in models.split(","):
        try:
            model_type = "sav"
            print(m)
            model,score = None, None
            if m == "basic_svc":
                from model_trainer import train_basic_svc
                model, score = train_basic_svc(x_train, x_test, y_train, y_test, sample_weights )
                feature_importances = dict(sorted([(x,y) for x,y in zip(model.coef_.tolist()[0], list(features))]))
                metadata["basic_svc"] = {
                    "feature_importance": dict(zip(feature_importances.values(), feature_importances.keys()))
                }
            elif m == "grid_svc":
                from model_trainer import train_grid_svc
                model, score, best_params = train_grid_svc(x_train, x_test, y_train, y_test, sample_weights )
                feature_importances = dict(sorted([(x,y) for x,y in zip(model.coef_.tolist()[0], list(features))]))
                metadata["grid_svc"] = {
                    "feature_importance": dict(zip(feature_importances.values(), feature_importances.keys())),
                    "selected_params": best_params
                }
            elif m == "poly_svc":
                from model_trainer import train_poly_svc
                model, score = train_poly_svc(x_train, x_test, y_train, y_test, c=3, sample_weight=sample_weights)
                metadata["poly_svc"] = {}
            elif m == "grid_poly":
                from model_trainer import train_grid_poly_svc
                model, score, selected_params = train_grid_poly_svc(x_train, x_test, y_train, y_test)
                metadata["grid_poly"] = {"selected_params": json.dumps(selected_params)}
            elif m == "basic_xgb":
                from model_trainer import train_xgb_basic
                model, score = train_xgb_basic(x_train, x_test, y_train, y_test, sample_weight=sample_weights)
                print(model)
                print(score)
                metadata["basic_xgb"] = {}
            elif m == "grid_xgb":
                from model_trainer import train_xgb_grid
                print("HERE")
                model, score, selected_params = train_xgb_grid(x_train, x_test, y_train, y_test, sample_weight=sample_weights)
                print(f"Grid XGB Params: {selected_params}")
                metadata["grid_xgb"] = {"selected_params": selected_params}
            elif m == "basic_dnn":
                from model_trainer import train_dnn
                model, score = train_dnn(x_train,x_test, y_train, y_test, sample_weight=sample_weights)
                model_type = "h5"
                metadata["basic_dnn"] = {}
                print(score)
                
            
            print(f"Model\t\tScore\n{m}\t{score}")
            trained_models.append((m, model, score, model_type))
        except Exception as ex:
            print(f"Failed {m}: {ex}")

    
    if save:
        run_id = str(uuid.uuid4())[0:8]
        print(f"Saving models in run: {run_id}")
        run_root = f"{MODELS_ROOT}/{run_id}"
        os.makedirs(run_root)
        for m_id, model, score, m_type in trained_models:
            metadata[m_id]["score"] = score
            if m_type == "sav":
                model_file_path = os.path.join(run_root, f"{m_id}.sav")
                pickle.dump(model, open(model_file_path, 'wb'))
            else:
                model_file_path = os.path.join(run_root, f"{m_id}.h5")
                model.save(model_file_path)
        if scale:
            pickle.dump(scaler, open(os.path.join(run_root, "scaler.pkl"), "wb"))
        
        if len(metadata.keys()) > 0:
            with open(os.path.join(run_root, "metadata.json"),"w") as wf:
                json.dump(metadata, wf)

    return

@click.command()
@click.option('--run', help="Run identifier")
@click.option('--models', default="basic_svc", help="Comma separated list of models to use")
@click.option('--seasons', default=CS_ALL_SEASONS, help="Comma separated list of seasons")
@click.option('--data', default=None, help="Training Data version")
def evaluate(run, models, seasons, data):
    """Evaluate model on a historical tournament
    """
    models = models.split(",")
    seasons = [int(s) for s in seasons.split(",")]
    run_root = os.path.join(MODELS_ROOT, run)
    scaler = get_scaler(run)
    
    results = {}
    # Load and Evaluate models
    for m in models:
        results[m] = {}
        model_path = os.path.join(run_root, f"{m}.sav")
        model = pickle.load(open(model_path,"rb"))
        for s in seasons:
            print()
            results[m][s] = evaluate_model_on_tournament(model, scaler, s, data)
    
    # Display Results
    for m, res in results.items():
        print(m)
        [print(f"{k} - {v['score']}") for k,v in res.items()]
    return

# Fill Kaggle Submission with Model
@click.command()
@click.option('-r', '--run-id', help="Run id from the model training")
@click.option('-m', '--model-ids', help = "Comma separated list of model ids")
@click.option('-s', '--submission-id', help="Identifier for the submission")
@click.option('--run-folder', help="Run Folder")
@click.option('--stage', default="2", help="Kaggle Stage")
def submit(run_id, model_ids, submission_id, run_folder, stage):
    if run_folder is None:
        run_root = os.path.join(MODELS_ROOT, run_id)
    else:
        run_root = os.path.join(run_folder, submission_id)
    submission_root = f"{SUBMISSIONS_ROOT}/{submission_id}"
    scaler_path = os.path.join(run_root, "scaler.pkl")
    metadata_path = os.path.join(run_root, "metadata.json")
    scaler = None #pickle.load(open(scaler_path, "rb"))
    for m in model_ids.split(","):
        model_path = os.path.join(run_root, f"{m}.sav")
        if os.path.exists(model_path):
            model_type = "sav"
            model = pickle.load(open(model_path,"rb"))
        elif(os.path.exists(model_path.replace("sav","h5"))):
            model_type = "h5"
            model_path = model_path.replace("sav","h5")
            model = keras.models.load_model(model_path)
        fill_submission(model, scaler, m, submission_id, stage=stage, data_dir="Training/ComputedV1", model_type=model_type)
        if run_folder is None:
            shutil.copy2(model_path, os.path.join(submission_root, f"{m}.sav"))
            shutil.copy2(scaler_path, os.path.join(submission_root, f"scaler.pkl"))
            shutil.copy2(metadata_path, os.path.join(submission_root, f"metadata.json"))
    return

@click.command()
@click.option('-s', '--submission-id', help="Identifier for the submission")
@click.option('-m', '--model-id', help = "Model id")
@click.option('-y', '--year')
def bracket(submission_id, model_id, year, sub_dir = "2022"):
    """Display bracket
    """
    year = int(year)
    if year == 2022:
        subPath = f'Results/{sub_dir}/{submission_id}/{model_id}_2.csv'
    else:
        subPath = f'Results/{sub_dir}/{submission_id}/{model_id}_1.csv'
    b = build_bracket(
        outputPath=f'Results/{sub_dir}/{submission_id}/{model_id}_{year}.png',
        teamsPath=f'{DATA_ROOT}/Stage2/MTeams.csv',
        seedsPath=f'{DATA_ROOT}/Stage2/MNCAATourneySeeds.csv',
        submissionPath=subPath,
        slotsPath=f'{DATA_ROOT}/Stage2/MNCAATourneySlots.csv',
        year=int(year)
    )

cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(submit)
cli.add_command(bracket)
if __name__ == "__main__":
    cli()