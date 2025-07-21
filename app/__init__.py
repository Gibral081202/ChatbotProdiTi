# This file makes the app directory a Python package 

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager  # type: ignore

db = SQLAlchemy()
login_manager = LoginManager()
# Note: Place custom static files (CSS/JS) in 'app/static/' for Flask.
# Flask will serve them automatically if the directory exists. 