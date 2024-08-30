"""Set server default for task.completed_at and task.assigned_at to none and add num_retries.

Revision ID: b9860676dd49
Revises: 5ad873484aa3
Create Date: 2024-08-22 07:54:55.921710

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9860676dd49"
down_revision: str | None = "5ad873484aa3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database to this revision from previous."""
    with op.batch_alter_table("tasks", schema=None) as batch_op:
        batch_op.alter_column(
            "completed_at",
            server_default=None,
        )
        batch_op.alter_column(
            "assigned_at",
            server_default=None,
        )
        batch_op.add_column(
            sa.Column(
                "num_retries",
                sa.Integer(),
                nullable=False,
                comment="Number of retries",
                server_default=sa.text("0"),
            )
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade database from this revision to previous."""
    with op.batch_alter_table("tasks", schema=None) as batch_op:
        batch_op.drop_column("num_retries")

    # ### end Alembic commands ###
