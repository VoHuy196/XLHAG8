#!/bin/bash
echo "Creating MLflow database..."
psql -v ON_ERROR_STOP=0 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE DATABASE mlflow;
EOSQL
echo "MLflow database setup completed"
