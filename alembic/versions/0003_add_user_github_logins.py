from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite, postgresql

revision = "0003_add_user_github_logins"
down_revision = "0002_add_user_is_admin"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # JSON is portable across SQLite and Postgres; the explicit dialect
    # variants keep each backend happy with its native JSON type.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        column_type = postgresql.JSONB(astext_type=sa.Text())
    else:
        column_type = sqlite.JSON()
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column(
                "github_logins",
                column_type,
                server_default="[]",
                nullable=False,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("github_logins")
