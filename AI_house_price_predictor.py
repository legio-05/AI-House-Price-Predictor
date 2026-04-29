#Portfolio AI Project: Advanced House Price Prediction from Scratch


#What it demonstrates:
    #1. Synthetic dataset generation with realistic housing features.
    #2. Feature engineering for interaction effects.
    #3. Train/test split and k-fold cross-validation.
    #4. Standard scaling learned only from training data.
    #5. Linear regression trained with gradient descent and L2 regularization.
    #6. MAE, RMSE, MAPE, R-squared, and baseline comparison.
    #7. Prediction intervals based on training residuals.
    #8. Feature contribution explanations.
    #9. JSON model persistence.

#Examples:
    #python AI_house_price_predictor.py
    #python AI_house_price_predictor.py --menu
    #python AI_house_price_predictor.py --demo
    #python AI_house_price_predictor.py --evaluate
    #python AI_house_price_predictor.py --predict --size 1800 --bedrooms 3 --bathrooms 2 --age 12 --distance 7 --quality 8 --garage 1
    #python AI_house_price_predictor.py --save house_price_model.json
    #python AI_house_price_predictor.py --load house_price_model.json --predict --size 2100 --bedrooms 4 --bathrooms 3 --age 5 --distance 4 --quality 9 --garage 2

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


RAW_FEATURES = [
    "size_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_km",
    "quality_score",
    "garage_spaces",
]

MODEL_FEATURES = [
    "size_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_km",
    "quality_score",
    "garage_spaces",
    "size_quality",
    "age_distance",
]

TARGET = "price_usd"
DEFAULT_DATASET_SIZE = 220

FEATURE_RANGES = {
    "size_sqft": (450, 5200),
    "bedrooms": (1, 7),
    "bathrooms": (1, 5),
    "age_years": (0, 80),
    "distance_km": (0, 35),
    "quality_score": (1, 10),
    "garage_spaces": (0, 4),
}

FEATURE_LABELS = {
    "size_sqft": "living area",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "age_years": "property age",
    "distance_km": "distance from center",
    "quality_score": "quality score",
    "garage_spaces": "garage spaces",
    "size_quality": "size x quality interaction",
    "age_distance": "age x distance interaction",
}


def generate_housing_data(count: int = DEFAULT_DATASET_SIZE, seed: int = 24) -> list[dict[str, float]]:
    """Create a reproducible housing dataset with noisy but learnable prices."""
    rng = random.Random(seed)
    rows = []

    for _ in range(count):
        bedrooms = rng.choices([1, 2, 3, 4, 5, 6], weights=[5, 16, 32, 27, 15, 5])[0]
        size_sqft = rng.gauss(620 + bedrooms * 430, 190)
        size_sqft = clamp(size_sqft, *FEATURE_RANGES["size_sqft"])

        bathrooms = clamp(round(bedrooms * 0.55 + rng.uniform(0.2, 1.6)), 1, 5)
        age_years = clamp(rng.gammavariate(2.0, 9.0), *FEATURE_RANGES["age_years"])
        distance_km = clamp(rng.gammavariate(2.2, 4.2), *FEATURE_RANGES["distance_km"])
        quality_score = clamp(round(rng.gauss(6.2, 1.6)), 1, 10)
        garage_spaces = clamp(round((size_sqft - 900) / 850 + rng.uniform(-0.3, 1.2)), 0, 4)

        renovation_bonus = 18_000 if age_years > 25 and quality_score >= 8 else 0
        central_bonus = max(0, 12 - distance_km) * 5_500
        price = (
            42_000
            + size_sqft * 142
            + bedrooms * 9_000
            + bathrooms * 24_000
            + garage_spaces * 18_000
            + quality_score * 31_000
            - age_years * 2_350
            - distance_km * 7_200
            + (size_sqft * quality_score) * 7.5
            - (age_years * distance_km) * 480
            + central_bonus
            + renovation_bonus
            + rng.gauss(0, 22_000)
        )

        rows.append(
            {
                "size_sqft": round(size_sqft),
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age_years": round(age_years, 1),
                "distance_km": round(distance_km, 1),
                "quality_score": quality_score,
                "garage_spaces": garage_spaces,
                "price_usd": max(75_000, round(price)),
            }
        )

    return rows


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_model_features(row: dict[str, float]) -> dict[str, float]:
    """Add interaction features that help a linear model capture richer patterns."""
    return {
        "size_sqft": row["size_sqft"],
        "bedrooms": row["bedrooms"],
        "bathrooms": row["bathrooms"],
        "age_years": row["age_years"],
        "distance_km": row["distance_km"],
        "quality_score": row["quality_score"],
        "garage_spaces": row["garage_spaces"],
        "size_quality": row["size_sqft"] * row["quality_score"],
        "age_distance": row["age_years"] * row["distance_km"],
    }


@dataclass
class StandardScaler:
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, feature_rows: list[dict[str, float]]) -> None:
        for feature in MODEL_FEATURES:
            values = [row[feature] for row in feature_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.means[feature] = mean
            self.stds[feature] = math.sqrt(variance) or 1.0

    def transform(self, feature_row: dict[str, float]) -> list[float]:
        return [(feature_row[feature] - self.means[feature]) / self.stds[feature] for feature in MODEL_FEATURES]


@dataclass
class LinearRegressionGD:
    """Multiple linear regression trained with batch gradient descent."""

    learning_rate: float = 0.035
    epochs: int = 7500
    l2_strength: float = 0.002
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    residual_std_usd: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def fit(self, rows: list[dict[str, float]]) -> None:
        feature_rows = [build_model_features(row) for row in rows]
        self.scaler.fit(feature_rows)

        x_train = [self.scaler.transform(feature_row) for feature_row in feature_rows]
        y_train = [row[TARGET] / 1000 for row in rows]
        self.weights = [0.0 for _ in MODEL_FEATURES]
        self.bias = sum(y_train) / len(y_train)
        self.loss_history = []

        for epoch in range(self.epochs):
            weight_gradients = [0.0 for _ in MODEL_FEATURES]
            bias_gradient = 0.0
            squared_error = 0.0

            for features, actual in zip(x_train, y_train):
                predicted = self._predict_scaled(features)
                error = predicted - actual
                squared_error += error**2
                bias_gradient += error
                for index, value in enumerate(features):
                    weight_gradients[index] += error * value

            n = len(rows)
            self.bias -= self.learning_rate * bias_gradient / n
            for index, weight in enumerate(self.weights):
                regularization = self.l2_strength * weight
                gradient = weight_gradients[index] / n + regularization
                self.weights[index] -= self.learning_rate * gradient

            if epoch % 500 == 0 or epoch == self.epochs - 1:
                self.loss_history.append(squared_error / n)

        residuals = [self.predict(row) - row[TARGET] for row in rows]
        self.residual_std_usd = math.sqrt(sum(error**2 for error in residuals) / len(residuals))

    def predict(self, row: dict[str, float]) -> float:
        feature_row = build_model_features(row)
        scaled_features = self.scaler.transform(feature_row)
        return self._predict_scaled(scaled_features) * 1000

    def prediction_interval(self, row: dict[str, float], z_score: float = 1.96) -> tuple[float, float, float]:
        prediction = self.predict(row)
        margin = z_score * self.residual_std_usd
        return prediction, prediction - margin, prediction + margin

    def explain(self, row: dict[str, float]) -> list[tuple[str, float]]:
        feature_row = build_model_features(row)
        scaled_features = self.scaler.transform(feature_row)
        contributions = [
            (feature, self.weights[index] * scaled_features[index] * 1000)
            for index, feature in enumerate(MODEL_FEATURES)
        ]
        return sorted(contributions, key=lambda item: abs(item[1]), reverse=True)

    def save(self, path: str | Path) -> None:
        model = {
            "model_type": "from_scratch_linear_regression",
            "features": MODEL_FEATURES,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2_strength": self.l2_strength,
            "weights": self.weights,
            "bias": self.bias,
            "means": self.scaler.means,
            "stds": self.scaler.stds,
            "residual_std_usd": self.residual_std_usd,
        }
        Path(path).write_text(json.dumps(model, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearRegressionGD":
        model_data = json.loads(Path(path).read_text(encoding="utf-8"))
        if model_data.get("features") != MODEL_FEATURES:
            raise ValueError("Saved model uses a different feature layout.")

        model = cls(
            learning_rate=model_data["learning_rate"],
            epochs=model_data["epochs"],
            l2_strength=model_data["l2_strength"],
            weights=model_data["weights"],
            bias=model_data["bias"],
            residual_std_usd=model_data["residual_std_usd"],
        )
        model.scaler = StandardScaler(means=model_data["means"], stds=model_data["stds"])
        return model

    def _predict_scaled(self, scaled_features: list[float]) -> float:
        return self.bias + sum(weight * value for weight, value in zip(self.weights, scaled_features))


def train_test_split(
    rows: list[dict[str, float]],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, round(len(rows) * test_ratio))
    return shuffled[test_size:], shuffled[:test_size]


def k_fold_cross_validate(rows: list[dict[str, float]], folds: int = 5, seed: int = 42) -> dict[str, float]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    fold_rows = [shuffled[index::folds] for index in range(folds)]
    metrics_by_fold = []

    for fold_index, test_rows in enumerate(fold_rows):
        train_rows = [
            row
            for other_index, fold in enumerate(fold_rows)
            if other_index != fold_index
            for row in fold
        ]
        model = LinearRegressionGD()
        model.fit(train_rows)
        metrics_by_fold.append(evaluate(model, test_rows))

    return average_metrics(metrics_by_fold)


def evaluate(model: LinearRegressionGD, rows: list[dict[str, float]]) -> dict[str, float]:
    actual_values = [row[TARGET] for row in rows]
    predictions = [model.predict(row) for row in rows]
    errors = [predicted - actual for predicted, actual in zip(predictions, actual_values)]

    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
    mape = sum(abs(error) / actual for error, actual in zip(errors, actual_values)) / len(errors) * 100
    mean_actual = sum(actual_values) / len(actual_values)
    total_variance = sum((actual - mean_actual) ** 2 for actual in actual_values)
    unexplained_variance = sum(error**2 for error in errors)
    r_squared = 1 - unexplained_variance / total_variance if total_variance else 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape, "r_squared": r_squared}


def baseline_metrics(train_rows: list[dict[str, float]], test_rows: list[dict[str, float]]) -> dict[str, float]:
    average_price = sum(row[TARGET] for row in train_rows) / len(train_rows)
    actual_values = [row[TARGET] for row in test_rows]
    errors = [average_price - actual for actual in actual_values]

    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
    mape = sum(abs(error) / actual for error, actual in zip(errors, actual_values)) / len(errors) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape, "r_squared": 0.0}


def average_metrics(metrics_by_fold: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(metrics[key] for metrics in metrics_by_fold) / len(metrics_by_fold)
        for key in metrics_by_fold[0]
    }


def train_default_model(rows: int = DEFAULT_DATASET_SIZE) -> tuple[LinearRegressionGD, list[dict[str, float]], list[dict[str, float]]]:
    dataset = generate_housing_data(rows)
    train_rows, test_rows = train_test_split(dataset)
    model = LinearRegressionGD()
    model.fit(train_rows)
    return model, train_rows, test_rows


def validate_house(row: dict[str, float]) -> None:
    errors = []
    for feature, (minimum, maximum) in FEATURE_RANGES.items():
        value = row[feature]
        if not minimum <= value <= maximum:
            errors.append(f"{feature} must be between {minimum} and {maximum}")

    if errors:
        raise SystemExit("Invalid input:\n  - " + "\n  - ".join(errors))


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  MAE: {format_money(metrics['mae'])}")
    print(f"  RMSE: {format_money(metrics['rmse'])}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R-squared: {metrics['r_squared']:.3f}")


def print_prediction(model: LinearRegressionGD, row: dict[str, float]) -> None:
    validate_house(row)
    prediction, low, high = model.prediction_interval(row)
    print(f"Predicted house price: {format_money(prediction)}")
    print(f"Approximate 95% range: {format_money(low)} to {format_money(high)}")
    print()
    print("Largest model contributions compared with an average house:")
    for feature, impact in model.explain(row)[:6]:
        direction = "adds" if impact >= 0 else "subtracts"
        print(f"  - {FEATURE_LABELS[feature]}: {direction} {format_money(abs(impact))}")


def run_demo(rows: int = DEFAULT_DATASET_SIZE) -> None:
    model, train_rows, test_rows = train_default_model(rows)
    train_metrics = evaluate(model, train_rows)
    test_metrics = evaluate(model, test_rows)
    baseline = baseline_metrics(train_rows, test_rows)
    cv_metrics = k_fold_cross_validate(generate_housing_data(rows), folds=5)
    improvement = (baseline["mae"] - test_metrics["mae"]) / baseline["mae"] * 100

    print("Advanced House Price Prediction from Scratch")
    print("=" * 45)
    print(f"Generated dataset rows: {rows}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Engineered model features: {len(MODEL_FEATURES)}")
    print()
    print_metrics("Test performance:", test_metrics)
    print()
    print_metrics("5-fold cross-validation average:", cv_metrics)
    print()
    print_metrics("Training performance:", train_metrics)
    print()
    print_metrics("Simple baseline performance:", baseline)
    print(f"Model improves test MAE over baseline by {improvement:.1f}%")
    print()

    example_house = {
        "size_sqft": 2200,
        "bedrooms": 4,
        "bathrooms": 3,
        "age_years": 8,
        "distance_km": 5,
        "quality_score": 8,
        "garage_spaces": 2,
    }
    print("Example prediction:")
    print_prediction(model, example_house)


def build_house_from_args(args: argparse.Namespace) -> dict[str, float]:
    missing = [
        flag
        for flag, value in {
            "--size": args.size,
            "--bedrooms": args.bedrooms,
            "--bathrooms": args.bathrooms,
            "--age": args.age,
            "--distance": args.distance,
            "--quality": args.quality,
            "--garage": args.garage,
        }.items()
        if value is None
    ]
    if missing:
        raise SystemExit(f"Missing values for prediction: {', '.join(missing)}")

    house = {
        "size_sqft": args.size,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "age_years": args.age,
        "distance_km": args.distance,
        "quality_score": args.quality,
        "garage_spaces": args.garage,
    }
    validate_house(house)
    return house


def prompt_number(prompt: str, feature: str, whole_number: bool = False) -> float:
    minimum, maximum = FEATURE_RANGES[feature]

    while True:
        raw_value = input(f"{prompt} ({minimum}-{maximum}): ").strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a number.")
            continue

        if not minimum <= value <= maximum:
            print(f"Value must be between {minimum} and {maximum}.")
            continue

        if whole_number and not value.is_integer():
            print("Please enter a whole number.")
            continue

        return int(value) if whole_number else value


def build_house_from_menu() -> dict[str, float]:
    print()
    print("Enter house details")
    print("-" * 19)
    return {
        "size_sqft": prompt_number("Size in square feet", "size_sqft"),
        "bedrooms": prompt_number("Bedrooms", "bedrooms", whole_number=True),
        "bathrooms": prompt_number("Bathrooms", "bathrooms", whole_number=True),
        "age_years": prompt_number("Age in years", "age_years"),
        "distance_km": prompt_number("Distance to city center in km", "distance_km"),
        "quality_score": prompt_number("Quality score", "quality_score", whole_number=True),
        "garage_spaces": prompt_number("Garage spaces", "garage_spaces", whole_number=True),
    }


def print_evaluation_report(
    model: LinearRegressionGD,
    train_rows: list[dict[str, float]],
    test_rows: list[dict[str, float]],
    rows: int,
) -> None:
    baseline = baseline_metrics(train_rows, test_rows)
    test_metrics = evaluate(model, test_rows)
    train_metrics = evaluate(model, train_rows)
    improvement = (baseline["mae"] - test_metrics["mae"]) / baseline["mae"] * 100

    print()
    print_metrics("Test performance:", test_metrics)
    print()
    print_metrics("Training performance:", train_metrics)
    print()
    print_metrics("Simple baseline performance:", baseline)
    print(f"Model improves test MAE over baseline by {improvement:.1f}%")
    print()
    print("Running 5-fold cross-validation. This may take a few seconds...")
    print_metrics("5-fold cross-validation average:", k_fold_cross_validate(generate_housing_data(rows)))


def run_menu(rows: int = DEFAULT_DATASET_SIZE) -> None:
    print("Training model for interactive menu...")
    model, train_rows, test_rows = train_default_model(rows)

    while True:
        print()
        print("House Price Predictor Menu")
        print("=" * 28)
        print("1. Predict a house price")
        print("2. Show model evaluation")
        print("3. Run full demo")
        print("4. Save trained model")
        print("5. Load model and predict")
        print("6. Exit")

        choice = input("Choose an option (1-6): ").strip().lower()

        if choice == "1":
            house = build_house_from_menu()
            print()
            print_prediction(model, house)
        elif choice == "2":
            print_evaluation_report(model, train_rows, test_rows, rows)
        elif choice == "3":
            print()
            run_demo(rows)
        elif choice == "4":
            path = input("Save path [house_price_model.json]: ").strip() or "house_price_model.json"
            model.save(path)
            print(f"Saved model to {path}")
        elif choice == "5":
            path = input("Model path [house_price_model.json]: ").strip() or "house_price_model.json"
            try:
                loaded_model = LinearRegressionGD.load(path)
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
                print(f"Could not load model: {error}")
                continue

            house = build_house_from_menu()
            print()
            print_prediction(loaded_model, house)
        elif choice in {"6", "q", "quit", "exit"}:
            print("Goodbye.")
            break
        else:
            print("Please choose a number from 1 to 6.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict house prices with an explainable from-scratch regression model."
    )
    parser.add_argument("--menu", action="store_true", help="open the interactive console menu")
    parser.add_argument("--demo", action="store_true", help="run the full portfolio demo")
    parser.add_argument("--evaluate", action="store_true", help="print train/test and cross-validation metrics")
    parser.add_argument("--predict", action="store_true", help="predict a house price")
    parser.add_argument("--size", type=float, help="house size in square feet")
    parser.add_argument("--bedrooms", type=float, help="number of bedrooms")
    parser.add_argument("--bathrooms", type=float, help="number of bathrooms")
    parser.add_argument("--age", type=float, help="house age in years")
    parser.add_argument("--distance", type=float, help="distance to city center in kilometers")
    parser.add_argument("--quality", type=float, help="quality score from 1 to 10")
    parser.add_argument("--garage", type=float, help="number of garage spaces")
    parser.add_argument("--rows", type=int, default=DEFAULT_DATASET_SIZE, help="synthetic dataset size")
    parser.add_argument("--save", metavar="PATH", help="train and save a model as JSON")
    parser.add_argument("--load", metavar="PATH", help="load a saved model")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.rows < 50:
        raise SystemExit("--rows must be at least 50 for a meaningful train/test split")

    has_action = any(value for key, value in vars(args).items() if key != "rows")

    if args.menu or not has_action:
        run_menu(args.rows)
        return

    if args.demo:
        run_demo(args.rows)
        return

    if args.load:
        model = LinearRegressionGD.load(args.load)
        train_rows = []
        test_rows = []
    else:
        model, train_rows, test_rows = train_default_model(args.rows)

    if args.evaluate:
        if args.load:
            print("Evaluation requires a freshly generated dataset, so omit --load.")
        else:
            print_metrics("Test performance:", evaluate(model, test_rows))
            print()
            print_metrics("5-fold cross-validation average:", k_fold_cross_validate(generate_housing_data(args.rows)))

    if args.predict:
        house = build_house_from_args(args)
        print_prediction(model, house)

    if args.save:
        model.save(args.save)
        print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
