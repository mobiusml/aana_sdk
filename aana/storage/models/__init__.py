# ruff: noqa: F401
# We need to import all db models here and, other than in the class definitions
# themselves, only import them from aana.models.db directly. The reason for
# this is the way SQLAlchemy's declarative base works. You can use forward
# references like `parent = reference("Parent", backreferences="child")`, but the
# forward reference needs to have been resolved before the first constructor
# is called so that SqlAlchemy "knows" about it.
# See:
# https://docs.pylonsproject.org/projects/pyramid_cookbook/en/latest/database/sqlalchemy.html#importing-all-sqlalchemy-models
# (even if not using Pyramid)

from aana.storage.models.base import BaseEntity
from aana.storage.models.caption import CaptionEntity
from aana.storage.models.media import MediaEntity, MediaType
from aana.storage.models.task import TaskEntity
from aana.storage.models.transcript import TranscriptEntity
from aana.storage.models.video import VideoEntity
