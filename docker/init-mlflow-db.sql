-- Отдельная база под MLflow в том же postgres.
-- Выполняется один раз при первой инициализации контейнера.
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
