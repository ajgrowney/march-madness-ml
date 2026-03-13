# TF.js Linear Regression Demo

Place a TF.js model directory containing `model.json` and shard files at `web/models/saved_linear_tfjs/`.

Serve the `web/` folder with a static server and open `index.html`.

Example (from repository root):

```bash
python -m http.server --directory web 8000
# then open http://localhost:8000
```

If you trained and exported the model using `models/linear_regression_tfjs.js`, copy the generated folder into `web/models/saved_linear_tfjs/` before serving.
