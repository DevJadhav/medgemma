-- PostgreSQL Initialization Script
-- Creates tables for MedAI Compass application

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Sessions table for tracking user sessions
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Audit logs for HIPAA compliance
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

-- Create index for audit log queries
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- PHI is encrypted at rest
    encrypted_data BYTEA,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Studies table (for imaging)
CREATE TABLE IF NOT EXISTS studies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    study_instance_uid VARCHAR(255) UNIQUE,
    study_date DATE,
    modality VARCHAR(50),
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Series table
CREATE TABLE IF NOT EXISTS series (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_id UUID REFERENCES studies(id),
    series_instance_uid VARCHAR(255) UNIQUE,
    series_number INTEGER,
    modality VARCHAR(50),
    description TEXT,
    instance_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_id UUID REFERENCES studies(id),
    agent_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(100),
    findings JSONB,
    confidence FLOAT,
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    session_id UUID REFERENCES sessions(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL, -- 'patient', 'agent', 'clinician'
    agent_name VARCHAR(100),
    content TEXT,
    triage_result JSONB,
    requires_review BOOLEAN DEFAULT false,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_studies_patient ON studies(patient_id);
CREATE INDEX idx_analysis_study ON analysis_results(study_id);

-- Row-level security policies for multi-tenant access
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- LangGraph Checkpoint Tables for Workflow State Persistence
-- =============================================================================

-- Checkpoints table for LangGraph state persistence
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    type VARCHAR(50),
    checkpoint JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Checkpoint writes for pending writes
CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    idx INTEGER NOT NULL,
    channel VARCHAR(255) NOT NULL,
    type VARCHAR(50),
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Checkpoint blobs for large data
CREATE TABLE IF NOT EXISTS langgraph_checkpoint_blobs (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    channel VARCHAR(255) NOT NULL,
    version VARCHAR(255) NOT NULL,
    type VARCHAR(50),
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- Indexes for checkpoint queries
CREATE INDEX idx_checkpoints_thread ON langgraph_checkpoints(thread_id);
CREATE INDEX idx_checkpoints_created ON langgraph_checkpoints(created_at);
CREATE INDEX idx_checkpoint_writes_thread ON langgraph_checkpoint_writes(thread_id);
CREATE INDEX idx_checkpoint_blobs_thread ON langgraph_checkpoint_blobs(thread_id);

-- =============================================================================
-- Conversation Persistence Tables for Multi-Instance Support
-- =============================================================================

-- Conversation state cache for quick retrieval
CREATE TABLE IF NOT EXISTS conversation_state (
    session_id VARCHAR(255) PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    state JSONB NOT NULL,
    messages JSONB DEFAULT '[]'::jsonb,
    context JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_conversation_state_patient ON conversation_state(patient_id);
CREATE INDEX idx_conversation_state_updated ON conversation_state(updated_at);

-- =============================================================================
-- Dataset Management Tables for Data Ingestion Pipeline
-- =============================================================================

-- Datasets table for tracking dataset metadata
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    source_url VARCHAR(500),
    access_type VARCHAR(50) DEFAULT 'open', -- open, credentialed, application
    version VARCHAR(50),
    description TEXT,
    total_files INTEGER DEFAULT 0,
    downloaded_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    size_bytes BIGINT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending', -- pending, downloading, downloaded, processing, ready, error
    local_path VARCHAR(500),
    minio_prefix VARCHAR(255),
    schema_info JSONB DEFAULT '{}'::jsonb, -- file structure, columns, etc.
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    downloaded_at TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Ingestion jobs table for tracking batch processing
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- download, process, validate, index
    celery_task_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed, cancelled
    priority INTEGER DEFAULT 5,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    progress_percent FLOAT DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    error_log JSONB DEFAULT '[]'::jsonb,
    result JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Dataset files table for tracking individual files
CREATE TABLE IF NOT EXISTS dataset_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    file_path VARCHAR(1000) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50), -- dicom, png, npy, csv, json, etc.
    file_size BIGINT DEFAULT 0,
    checksum VARCHAR(64), -- SHA-256
    minio_object_name VARCHAR(500),
    processed BOOLEAN DEFAULT false,
    processing_error TEXT,
    metadata JSONB DEFAULT '{}'::jsonb, -- DICOM metadata, labels, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for dataset queries
CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_ingestion_jobs_dataset ON ingestion_jobs(dataset_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);
CREATE INDEX idx_ingestion_jobs_celery ON ingestion_jobs(celery_task_id);
CREATE INDEX idx_dataset_files_dataset ON dataset_files(dataset_id);
CREATE INDEX idx_dataset_files_processed ON dataset_files(processed);

-- Insert default dataset records
INSERT INTO datasets (name, source_url, access_type, description, status) VALUES
    ('synthea', 'https://synthetichealth.github.io/synthea/', 'open', 'Synthetic patient data generator', 'pending'),
    ('medquad', 'https://github.com/abachaa/MedQuAD', 'open', 'Medical Question Answering Dataset - 47K QA pairs', 'pending'),
    ('meddialog', 'https://github.com/UCSD-AI4H/Medical-Dialogue-System', 'open', 'Medical dialogue dataset', 'pending'),
    ('chestxray14', 'https://nihcc.app.box.com/v/ChestXray-NIHCC', 'open', 'NIH ChestX-ray14 - 112K images', 'pending'),
    ('lidc_idri', 'https://www.cancerimagingarchive.net/collection/lidc-idri/', 'open', 'LIDC-IDRI lung CT scans', 'pending'),
    ('mimic_cxr', 'https://physionet.org/content/mimic-cxr/', 'credentialed', 'MIMIC-CXR chest X-rays', 'pending'),
    ('mimic_iv', 'https://physionet.org/content/mimiciv/3.1/', 'credentialed', 'MIMIC-IV EHR data', 'pending'),
    ('camelyon16', 'https://camelyon16.grand-challenge.org/', 'open', 'CAMELYON16 pathology WSI', 'pending'),
    ('n2c2', 'https://n2c2.dbmi.hms.harvard.edu/data-sets', 'credentialed', 'n2c2 NLP datasets', 'pending')
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Escalations Table for Clinician Review Queue
-- =============================================================================

-- Escalations table for human review queue
CREATE TABLE IF NOT EXISTS escalations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) NOT NULL,
    patient_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reason VARCHAR(50) NOT NULL, -- critical_finding, low_confidence, safety_concern, manual_request
    priority VARCHAR(20) NOT NULL DEFAULT 'medium', -- high, medium, low
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, in_review, approved, rejected
    
    -- Source data
    original_message TEXT,
    diagnostic_result JSONB,
    communication_result JSONB,
    workflow_result JSONB,
    
    -- Context
    agent_type VARCHAR(50), -- diagnostic, communication, workflow
    confidence_score FLOAT,
    uncertainty_score FLOAT,
    
    -- Review fields
    assigned_to VARCHAR(255),
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    review_notes TEXT,
    modified_response TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX idx_escalations_status ON escalations(status);
CREATE INDEX idx_escalations_priority ON escalations(priority);
CREATE INDEX idx_escalations_reason ON escalations(reason);
CREATE INDEX idx_escalations_timestamp ON escalations(timestamp);
CREATE INDEX idx_escalations_patient ON escalations(patient_id);
CREATE INDEX idx_escalations_request ON escalations(request_id);
CREATE INDEX idx_escalations_assigned ON escalations(assigned_to);

-- Compound index for common query pattern
CREATE INDEX idx_escalations_pending_priority ON escalations(status, priority, timestamp)
    WHERE status IN ('pending', 'in_review');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO medai;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO medai;
