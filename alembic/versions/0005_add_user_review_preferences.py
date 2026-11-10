from alembic import op
import sqlalchemy as sa

revision = "0005_add_user_review_preferences"
down_revision = "0004_create_split_plans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "strict",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "enable_pr_review",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "strict")
    op.drop_column("users", "enable_pr_review")