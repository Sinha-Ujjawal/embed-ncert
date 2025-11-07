"""create table agent_conversation_history

Revision ID: 3b1bb18e45c9
Revises:
Create Date: 2025-11-04 10:15:02.645870

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '3b1bb18e45c9'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'agent_conversation_history',
        sa.Column('thread_id', sa.VARCHAR(64), nullable=False),
        sa.Column('mlflow_run_id', sa.VARCHAR(64), nullable=False),
        sa.Column('subject', sa.VARCHAR(64), nullable=False),
        sa.Column('timestamp_utc', sa.DATETIME, nullable=False),
        sa.Column('chunk_id', sa.INTEGER, nullable=False),
        sa.Column('model_repr', sa.VARCHAR(64), nullable=False),
        sa.Column('messages', sa.BLOB, nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('agent_conversation_history')
