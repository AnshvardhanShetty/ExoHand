import Database from "better-sqlite3";
import path from "path";
import fs from "fs";

const DB_PATH = path.join(__dirname, "..", "..", "exohand.db");

let db: Database.Database;

export function getDb(): Database.Database {
  if (!db) {
    db = new Database(DB_PATH);
    db.pragma("journal_mode = WAL");
    db.pragma("foreign_keys = ON");
    migrate(db);
  }
  return db;
}

function migrate(db: Database.Database) {
  const schemaPath = path.join(__dirname, "schema.sql");
  const schema = fs.readFileSync(schemaPath, "utf-8");
  db.exec(schema);

  // Add columns if missing (existing DBs)
  const cols = db.prepare("PRAGMA table_info(patients)").all() as any[];
  const colNames = cols.map((c: any) => c.name);
  if (!colNames.includes("description")) {
    db.exec("ALTER TABLE patients ADD COLUMN description TEXT NOT NULL DEFAULT ''");
  }
  if (!colNames.includes("dob")) {
    db.exec("ALTER TABLE patients ADD COLUMN dob TEXT NOT NULL DEFAULT ''");
  }
  if (!colNames.includes("hospital")) {
    db.exec("ALTER TABLE patients ADD COLUMN hospital TEXT NOT NULL DEFAULT ''");
  }

  // Add exercise_duration to sessions if missing
  const sessionCols = db.prepare("PRAGMA table_info(sessions)").all() as any[];
  const sessionColNames = sessionCols.map((c: any) => c.name);
  if (!sessionColNames.includes("exercise_duration")) {
    db.exec("ALTER TABLE sessions ADD COLUMN exercise_duration INTEGER");
  }
  if (!sessionColNames.includes("exercise_type")) {
    db.exec("ALTER TABLE sessions ADD COLUMN exercise_type TEXT");
  }

  // Clear old test data in development only (child tables first for FK compliance)
  if (process.env.NODE_ENV !== "production") {
    db.exec("DELETE FROM safety_events");
    db.exec("DELETE FROM metrics");
    db.exec("DELETE FROM reps");
    db.exec("DELETE FROM recommendations");
    db.exec("DELETE FROM sessions");
  }

  // Update seed data
  db.prepare("UPDATE therapists SET name = ? WHERE pin = '9999'").run("Dr. Shetty");
}
