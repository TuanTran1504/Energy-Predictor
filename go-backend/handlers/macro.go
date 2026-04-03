package handlers

import (
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
)

type MacroEvent struct {
	EventDate   string   `json:"event_date"`
	EventType   string   `json:"event_type"`
	Description string   `json:"description"`
	Actual      string   `json:"actual"`
	BtcImpact   *float64 `json:"btc_impact_24h"`
	EthImpact   *float64 `json:"eth_impact_24h"`
}

type FundingRate struct {
	Symbol  string  `json:"symbol"`
	Date    string  `json:"date"`
	RateAvg float64 `json:"rate_avg"`
}

type FearGreedEntry struct {
	Value          int    `json:"value"`
	Classification string `json:"classification"`
	Date           string `json:"date"`
}

type MacroResponse struct {
	FedRate      *MacroEvent      `json:"fed_rate"`
	CPI          *MacroEvent      `json:"cpi"`
	NFP          *MacroEvent      `json:"nfp"`
	FearGreed    *FearGreedEntry  `json:"fear_greed"`
	FearGreed30d []FearGreedEntry `json:"fear_greed_30d"`
	FundingBTC   *FundingRate     `json:"funding_btc"`
	FundingETH   *FundingRate     `json:"funding_eth"`
	RecentEvents []MacroEvent     `json:"recent_events"`
	FetchedAt    string           `json:"fetched_at"`
}

func GetMacroIndicators(c *gin.Context) {
	db, err := getDB()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer db.Close()

	result := MacroResponse{
		FetchedAt:    time.Now().UTC().Format(time.RFC3339),
		RecentEvents: []MacroEvent{},
		FearGreed30d: []FearGreedEntry{},
	}

	// ── Latest macro event per type ───────────────────────────────────────────
	for _, evType := range []string{"FED_RATE", "CPI", "NFP"} {
		row := db.QueryRow(`
			SELECT event_date, event_type, description, actual,
			       btc_impact_24h, eth_impact_24h
			FROM macro_events WHERE event_type = $1
			ORDER BY event_date DESC LIMIT 1
		`, evType)
		ev := &MacroEvent{}
		var evDate time.Time
		var actual sql.NullString
		var btcImpact, ethImpact sql.NullFloat64
		err := row.Scan(&evDate, &ev.EventType, &ev.Description, &actual, &btcImpact, &ethImpact)
		if err != nil {
			continue
		}
		ev.EventDate = evDate.UTC().Format("2006-01-02")
		if actual.Valid {
			ev.Actual = actual.String
		}
		if btcImpact.Valid {
			v := btcImpact.Float64
			ev.BtcImpact = &v
		}
		if ethImpact.Valid {
			v := ethImpact.Float64
			ev.EthImpact = &v
		}
		switch evType {
		case "FED_RATE":
			result.FedRate = ev
		case "CPI":
			result.CPI = ev
		case "NFP":
			result.NFP = ev
		}
	}

	// ── Fear & Greed — latest + 30d history ──────────────────────────────────
	rows, err := db.Query(`
		SELECT date, value FROM fear_greed_index
		ORDER BY date DESC LIMIT 30
	`)
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var d time.Time
			var v float64
			if rows.Scan(&d, &v) == nil {
				fg := FearGreedEntry{
					Value:          int(v),
					Classification: fearGreedClass(int(v)),
					Date:           d.Format("2006-01-02"),
				}
				if result.FearGreed == nil {
					cp := fg
					result.FearGreed = &cp
				}
				result.FearGreed30d = append(result.FearGreed30d, fg)
			}
		}
	}

	// Fallback: fetch live from alternative.me if DB is empty
	if result.FearGreed == nil {
		result.FearGreed = fetchLiveFearGreed()
	}

	// ── Funding rates — latest per symbol ────────────────────────────────────
	for _, sym := range []string{"BTC", "ETH"} {
		row := db.QueryRow(`
			SELECT symbol, date, rate_avg FROM funding_rates
			WHERE symbol = $1 ORDER BY date DESC LIMIT 1
		`, sym)
		fr := &FundingRate{}
		var d time.Time
		if row.Scan(&fr.Symbol, &d, &fr.RateAvg) == nil {
			fr.Date = d.Format("2006-01-02")
			if sym == "BTC" {
				result.FundingBTC = fr
			} else {
				result.FundingETH = fr
			}
		}
	}

	// ── Recent events timeline ────────────────────────────────────────────────
	rows2, err := db.Query(`
		SELECT event_date, event_type, description, actual,
		       btc_impact_24h, eth_impact_24h
		FROM macro_events ORDER BY event_date DESC LIMIT 20
	`)
	if err == nil {
		defer rows2.Close()
		for rows2.Next() {
			ev := MacroEvent{}
			var evDate time.Time
			var actual sql.NullString
			var btcImpact, ethImpact sql.NullFloat64
			if rows2.Scan(&evDate, &ev.EventType, &ev.Description, &actual, &btcImpact, &ethImpact) == nil {
				ev.EventDate = evDate.UTC().Format("2006-01-02")
				if actual.Valid {
					ev.Actual = actual.String
				}
				if btcImpact.Valid {
					v := btcImpact.Float64
					ev.BtcImpact = &v
				}
				if ethImpact.Valid {
					v := ethImpact.Float64
					ev.EthImpact = &v
				}
				result.RecentEvents = append(result.RecentEvents, ev)
			}
		}
	}

	c.JSON(http.StatusOK, result)
}

func fearGreedClass(v int) string {
	switch {
	case v <= 25:
		return "Extreme Fear"
	case v <= 45:
		return "Fear"
	case v <= 55:
		return "Neutral"
	case v <= 75:
		return "Greed"
	default:
		return "Extreme Greed"
	}
}

func fetchLiveFearGreed() *FearGreedEntry {
	resp, err := http.Get("https://api.alternative.me/fng/?limit=1&format=json")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var out struct {
		Data []struct {
			Value               string `json:"value"`
			ValueClassification string `json:"value_classification"`
			Timestamp           string `json:"timestamp"`
		} `json:"data"`
	}
	if json.Unmarshal(body, &out) != nil || len(out.Data) == 0 {
		return nil
	}
	v, _ := strconv.Atoi(out.Data[0].Value)
	ts, _ := strconv.ParseInt(out.Data[0].Timestamp, 10, 64)
	date := time.Unix(ts, 0).UTC().Format("2006-01-02")
	return &FearGreedEntry{
		Value:          v,
		Classification: out.Data[0].ValueClassification,
		Date:           date,
	}
}
