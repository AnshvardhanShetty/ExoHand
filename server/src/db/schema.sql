CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  pin TEXT NOT NULL UNIQUE,
  assist_level INTEGER NOT NULL DEFAULT 3,
  target_closure INTEGER NOT NULL DEFAULT 50,
  hold_duration REAL NOT NULL DEFAULT 2.0,
  rep_count INTEGER NOT NULL DEFAULT 10,
  description TEXT NOT NULL DEFAULT '',
  dob TEXT NOT NULL DEFAULT '',
  hospital TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS therapists (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  pin TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  ended_at TEXT,
  overall_score REAL,
  completion_rate REAL,
  avg_stability REAL,
  avg_accuracy REAL,
  exercise_duration INTEGER
);

CREATE TABLE IF NOT EXISTS reps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  rep_number INTEGER NOT NULL,
  accuracy REAL NOT NULL DEFAULT 0,
  stability REAL NOT NULL DEFAULT 0,
  time_to_target REAL NOT NULL DEFAULT 0,
  success INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  timestamp TEXT NOT NULL DEFAULT (datetime('now')),
  grip REAL,
  state_cmd TEXT,
  classifier_confidence REAL,
  assist_strength REAL
);

CREATE TABLE IF NOT EXISTS recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  type TEXT NOT NULL,
  message TEXT NOT NULL,
  approved INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS safety_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER REFERENCES sessions(id),
  timestamp TEXT NOT NULL DEFAULT (datetime('now')),
  event_type TEXT NOT NULL,
  details TEXT
);

CREATE TABLE IF NOT EXISTS exercise_programmes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  exercises TEXT NOT NULL DEFAULT '[]',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Seed default therapist for development
INSERT OR IGNORE INTO therapists (id, name, pin) VALUES (1, 'Dr. Shetty', '9999');
