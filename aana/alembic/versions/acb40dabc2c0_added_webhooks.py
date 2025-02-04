"""Added webhooks.

Revision ID: acb40dabc2c0
Revises: d40eba8ebc4c
Create Date: 2025-01-30 14:32:16.596842

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "acb40dabc2c0"
down_revision: str | None = "d40eba8ebc4c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database to this revision from previous."""
    # fmt: off
    op.create_table('webhooks',
    sa.Column('id', sa.UUID(), nullable=False, comment='Webhook ID'),
    sa.Column('user_id', sa.String(), nullable=True, comment='The user ID associated with the webhook'),
    sa.Column('url', sa.String(), nullable=False, comment='The URL to which the webhook will send requests'),
    sa.Column('events', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False, comment='List of events the webhook is subscribed to. If None, the webhook is subscribed to all events.'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False, comment='Timestamp when row is inserted'),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False, comment='Timestamp when row is updated'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_webhooks'))
    )
    with op.batch_alter_table('webhooks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_webhooks_user_id'), ['user_id'], unique=False)
    # fmt: on


def downgrade() -> None:
    """Downgrade database from this revision to previous."""
    with op.batch_alter_table("webhooks", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_webhooks_user_id"))

    op.drop_table("webhooks")
