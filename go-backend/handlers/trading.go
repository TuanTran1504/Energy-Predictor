package handlers

import (
	"database/sql"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq"
)

func getDB() (*sql.DB, error) {
	dsn := os.Getenv("DATABASE_URL")
	if !strings.Contains(dsn, "sslmode") {
		if strings.Contains(dsn, "?") {
			dsn += "&sslmode=require"
		} else {
			dsn += "?sslmode=require"
		}
	}
	return sql.Open("postgres", dsn)
}

func TradingStatus(c *gin.Context) {
	db, err := getDB()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer db.Close()

	// Open trades
	rows, err := db.Query(`
		SELECT id, symbol, side, entry_price, quantity, leverage,
		       stop_loss, take_profit, confidence, horizon,
		       binance_order_id, opened_at
		FROM trades WHERE status = 'OPEN'
		ORDER BY opened_at DESC
	`)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer rows.Close()

	type OpenTrade struct {
		ID             int64   `json:"id"`
		Symbol         string  `json:"symbol"`
		Side           string  `json:"side"`
		EntryPrice     float64 `json:"entry_price"`
		Quantity       float64 `json:"quantity"`
		Leverage       int     `json:"leverage"`
		StopLoss       float64 `json:"stop_loss"`
		TakeProfit     float64 `json:"take_profit"`
		Confidence     float64 `json:"confidence"`
		Horizon        int     `json:"horizon"`
		BinanceOrderID string  `json:"binance_order_id"`
		OpenedAt       string  `json:"opened_at"`
	}

	var openTrades []OpenTrade
	for rows.Next() {
		var t OpenTrade
		var orderID sql.NullString
		var openedAt sql.NullTime
		err := rows.Scan(
			&t.ID, &t.Symbol, &t.Side, &t.EntryPrice, &t.Quantity,
			&t.Leverage, &t.StopLoss, &t.TakeProfit, &t.Confidence,
			&t.Horizon, &orderID, &openedAt,
		)
		if err != nil {
			continue
		}
		if orderID.Valid {
			t.BinanceOrderID = orderID.String
		}
		if openedAt.Valid {
			t.OpenedAt = openedAt.Time.UTC().Format("2006-01-02T15:04:05Z")
		}
		openTrades = append(openTrades, t)
	}
	if openTrades == nil {
		openTrades = []OpenTrade{}
	}

	// Summary stats
	var totalTrades, wins int
	var totalPnl float64
	db.QueryRow(`
		SELECT COUNT(*), COALESCE(SUM(pnl_usdt),0),
		       COUNT(CASE WHEN pnl_usdt > 0 THEN 1 END)
		FROM trades WHERE status = 'CLOSED'
	`).Scan(&totalTrades, &totalPnl, &wins)

	winRate := 0.0
	if totalTrades > 0 {
		winRate = float64(wins) / float64(totalTrades) * 100
	}

	// Latest LLM decision
	var lastDecision sql.NullString
	var lastDecidedAt sql.NullTime
	db.QueryRow(`
		SELECT decision::text, decided_at FROM llm_decisions
		ORDER BY decided_at DESC LIMIT 1
	`).Scan(&lastDecision, &lastDecidedAt)

	lastDecisionStr := ""
	lastDecidedAtStr := ""
	if lastDecision.Valid {
		lastDecisionStr = lastDecision.String
	}
	if lastDecidedAt.Valid {
		lastDecidedAtStr = lastDecidedAt.Time.UTC().Format("2006-01-02T15:04:05Z")
	}

	c.JSON(http.StatusOK, gin.H{
		"open_trades":      openTrades,
		"total_closed":     totalTrades,
		"total_pnl":        totalPnl,
		"win_rate":         winRate,
		"wins":             wins,
		"last_decision":    lastDecisionStr,
		"last_decided_at":  lastDecidedAtStr,
	})
}

func TradingHistory(c *gin.Context) {
	db, err := getDB()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer db.Close()

	rows, err := db.Query(`
		SELECT id, symbol, side, status, entry_price, exit_price,
		       quantity, leverage, pnl_usdt, pnl_pct, confidence,
		       horizon, close_reason, opened_at, closed_at
		FROM trades ORDER BY opened_at DESC LIMIT 100
	`)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer rows.Close()

	type Trade struct {
		ID          int64    `json:"id"`
		Symbol      string   `json:"symbol"`
		Side        string   `json:"side"`
		Status      string   `json:"status"`
		EntryPrice  float64  `json:"entry_price"`
		ExitPrice   *float64 `json:"exit_price"`
		Quantity    float64  `json:"quantity"`
		Leverage    int      `json:"leverage"`
		PnlUsdt     *float64 `json:"pnl_usdt"`
		PnlPct      *float64 `json:"pnl_pct"`
		Confidence  float64  `json:"confidence"`
		Horizon     int      `json:"horizon"`
		CloseReason *string  `json:"close_reason"`
		OpenedAt    string   `json:"opened_at"`
		ClosedAt    *string  `json:"closed_at"`
	}

	var trades []Trade
	for rows.Next() {
		var t Trade
		var exitPrice, pnlUsdt, pnlPct sql.NullFloat64
		var closeReason sql.NullString
		var openedAt sql.NullTime
		var closedAt sql.NullTime

		err := rows.Scan(
			&t.ID, &t.Symbol, &t.Side, &t.Status,
			&t.EntryPrice, &exitPrice, &t.Quantity, &t.Leverage,
			&pnlUsdt, &pnlPct, &t.Confidence, &t.Horizon,
			&closeReason, &openedAt, &closedAt,
		)
		if err != nil {
			continue
		}
		if exitPrice.Valid   { t.ExitPrice   = &exitPrice.Float64 }
		if pnlUsdt.Valid     { t.PnlUsdt     = &pnlUsdt.Float64 }
		if pnlPct.Valid      { t.PnlPct      = &pnlPct.Float64 }
		if closeReason.Valid { t.CloseReason = &closeReason.String }
		if openedAt.Valid    { s := openedAt.Time.UTC().Format("2006-01-02T15:04:05Z"); t.OpenedAt = s }
		if closedAt.Valid    { s := closedAt.Time.UTC().Format("2006-01-02T15:04:05Z"); t.ClosedAt = &s }

		trades = append(trades, t)
	}
	if trades == nil {
		trades = []Trade{}
	}

	c.JSON(http.StatusOK, gin.H{"trades": trades})
}
