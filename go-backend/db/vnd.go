package db

import (
	"database/sql"
	"fmt"
	"os"
	"time"

	_ "github.com/lib/pq"
)

// GetDB returns a Postgres connection to Supabase
func GetDB() (*sql.DB, error) {
	connStr := os.Getenv("DATABASE_URL")
	if connStr == "" {
		return nil, fmt.Errorf("DATABASE_URL not set")
	}
	return sql.Open("postgres", connStr)
}

func SaveVNDRate(rate float64) error {
	db, err := GetDB()
	if err != nil {
		return err
	}
	defer db.Close()

	// Only save if last save was more than 30 minutes ago
	// Prevents duplicate rows from manual endpoint calls
	var lastSaved time.Time
	err = db.QueryRow(`
        SELECT fetched_at FROM vnd_rates
        ORDER BY fetched_at DESC
        LIMIT 1
    `).Scan(&lastSaved)

	if err == nil {
		// A row exists — check if it's recent
		if time.Since(lastSaved) < 30*time.Minute {
			return nil // skip — too recent
		}
	}

	_, err = db.Exec(
		"INSERT INTO vnd_rates (usd_to_vnd) VALUES ($1)",
		rate,
	)
	return err
}

// GetRateDaysAgo returns the closest stored rate N days ago
func GetRateDaysAgo(days int) (float64, error) {
	db, err := GetDB()
	if err != nil {
		return 0, err
	}
	defer db.Close()

	target := time.Now().UTC().AddDate(0, 0, -days)

	var rate float64
	err = db.QueryRow(`
        SELECT usd_to_vnd
        FROM vnd_rates
        WHERE fetched_at <= $1
        ORDER BY fetched_at DESC
        LIMIT 1
    `, target).Scan(&rate)

	if err == sql.ErrNoRows {
		return 0, fmt.Errorf("no rate found for %d days ago", days)
	}

	return rate, err
}
