-- Exact columns as specified (no extras).
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    creation_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    update_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations (conversation_id) ON DELETE CASCADE,
    creation_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    update_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_token_ct INTEGER NOT NULL DEFAULT 0,
    output_token_ct INTEGER NOT NULL DEFAULT 0,
    cost NUMERIC(12, 8) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS steps (
    step_id UUID PRIMARY KEY,
    turn_id UUID NOT NULL REFERENCES turns (turn_id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES conversations (conversation_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    step_type TEXT NOT NULL CHECK (step_type IN ('llm_call', 'tool_call', 'final_render')),
    input JSONB,
    output JSONB
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation_id ON turns (conversation_id);
CREATE INDEX IF NOT EXISTS idx_steps_turn_id ON steps (turn_id);
CREATE INDEX IF NOT EXISTS idx_steps_conversation_id ON steps (conversation_id);
