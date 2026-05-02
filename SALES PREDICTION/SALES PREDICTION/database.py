import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def build_database_url():
    db_name = os.getenv('DB_NAME', 'sales_forecasting_db')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_engine = os.getenv('DB_ENGINE', 'sqlite').lower()

    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        if db_engine in ('postgres', 'postgresql'):
            database_url = f"postgresql+pg8000://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        elif db_engine == 'sqlite':
            sqlite_path = resolve_sqlite_path()
            database_url = f"sqlite:///{sqlite_path}"
        else:
            sqlite_path = resolve_sqlite_path()
            database_url = f"sqlite:///{sqlite_path}"

    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql+pg8000://', 1)
    elif database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+pg8000://', 1)
    return database_url


def resolve_sqlite_path():
    """
    Prefer a stable instance directory to reduce file lock/disk I/O issues
    in synced folders, while preserving legacy DB if already present.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_path = os.path.join(base_dir, 'sales_forecasting.db')
    instance_dir = os.path.join(base_dir, 'instance')
    os.makedirs(instance_dir, exist_ok=True)
    instance_path = os.path.join(instance_dir, 'sales_forecasting.db')
    return legacy_path if os.path.exists(legacy_path) else instance_path


def init_database(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = build_database_url()
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite:///'):
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'connect_args': {
                'timeout': 30,
                'check_same_thread': False
            }
        }
    db.init_app(app)
    with app.app_context():
        try:
            db.create_all()
        except Exception:
            # Fallback to local SQLite when remote DB is unavailable.
            sqlite_path = resolve_sqlite_path()
            app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{sqlite_path}"
            db.engines.clear()
            db.create_all()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    rate = db.Column(db.Float, nullable=False)
    sales_first = db.Column(db.Float, nullable=False)
    sales_second = db.Column(db.Float, nullable=False)
    predicted_sales = db.Column(db.Float, nullable=False)
    cost = db.Column(db.Float, nullable=True) # New field for P/L
    actual_sales = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class SalesRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    product = db.Column(db.String(120), nullable=False, index=True)
    region = db.Column(db.String(120), nullable=False, index=True)
    season = db.Column(db.String(40), nullable=False)
    festival_name = db.Column(db.String(80), nullable=False, default='None')
    is_festival_day = db.Column(db.Boolean, nullable=False, default=False)
    discount_percentage = db.Column(db.Float, nullable=False, default=0.0)
    sales_amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
