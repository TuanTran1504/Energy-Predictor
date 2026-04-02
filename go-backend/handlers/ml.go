package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

// MLPrediction mirrors PredictionOut from the Python service.
type MLPrediction struct {
	Symbol      string  `json:"symbol"`
	Direction   string  `json:"direction"`
	Confidence  float64 `json:"confidence"`
	UpProb      float64 `json:"up_prob"`
	DownProb    float64 `json:"down_prob"`
	PredictedAt string  `json:"predicted_at"`
}

func pythonMLURL() string {
	if u := os.Getenv("PYTHON_ML_URL"); u != "" {
		return u
	}
	return "http://localhost:8001"
}

// MLBacktest proxies GET /ml/backtest → Python GET /backtest
func MLBacktest(c *gin.Context) {
	symbol := c.Query("symbol")
	days := c.Query("days")
	lookahead := c.Query("lookahead")

	url := pythonMLURL() + "/backtest?symbol=" + symbol
	if days != "" {
		url += "&days=" + days
	}
	if lookahead != "" {
		url += "&lookahead=" + lookahead
	}

	client := &http.Client{Timeout: 45 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "ML service unavailable: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read ML response"})
		return
	}

	c.Data(resp.StatusCode, "application/json", body)
}

// FetchMLPredictions calls Python GET /predict/live/all for BTC and ETH concurrently.
// One DB read per symbol covers all horizons — returns "BTC_1d", "BTC_7d", "ETH_1d", "ETH_7d".
func FetchMLPredictions() map[string]*MLPrediction {
	type result struct {
		preds map[string]*MLPrediction
		err   error
	}
	ch := make(chan result, 2)

	for _, sym := range []string{"BTC", "ETH"} {
		go func(s string) {
			preds, err := fetchAllPredictions(s)
			ch <- result{preds, err}
		}(sym)
	}

	predictions := make(map[string]*MLPrediction)
	for i := 0; i < 2; i++ {
		r := <-ch
		if r.err != nil {
			fmt.Printf("Warning: ML predictions failed: %v\n", r.err)
			continue
		}
		for k, v := range r.preds {
			predictions[k] = v
		}
	}
	return predictions
}

func fetchAllPredictions(symbol string) (map[string]*MLPrediction, error) {
	url := pythonMLURL() + "/predict/live/all?symbol=" + symbol
	client := &http.Client{Timeout: 30 * time.Second}

	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("python returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var raw map[string]MLPrediction
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}

	result := make(map[string]*MLPrediction, len(raw))
	for k, v := range raw {
		v := v
		result[k] = &v
	}
	return result, nil
}

// MLAnalyze proxies POST /ml/analyze → Python POST /analyze (streaming)
func MLAnalyze(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	url := pythonMLURL() + "/analyze"
	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest("POST", url, io.NopCloser(
		func() interface{ Read([]byte) (int, error) } {
			return &byteReader{data: body, pos: 0}
		}(),
	))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "ML service unavailable: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	c.Header("Content-Type", "text/plain; charset=utf-8")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Status(resp.StatusCode)
	io.Copy(c.Writer, resp.Body)
}

type byteReader struct {
	data []byte
	pos  int
}

func (b *byteReader) Read(p []byte) (int, error) {
	if b.pos >= len(b.data) {
		return 0, io.EOF
	}
	n := copy(p, b.data[b.pos:])
	b.pos += n
	return n, nil
}

// MLTradeChat proxies POST /ml/trade/chat → Python POST /trade/chat (streaming)
func MLTradeChat(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}
	url    := pythonMLURL() + "/trade/chat"
	client := &http.Client{Timeout: 120 * time.Second}
	req, _ := http.NewRequest("POST", url, &byteReader{data: body})
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()
	c.Header("Content-Type", "text/plain; charset=utf-8")
	c.Status(resp.StatusCode)
	io.Copy(c.Writer, resp.Body)
}

// MLTradeExecute proxies POST /ml/trade/execute → Python POST /trade/execute
func MLTradeExecute(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}
	url    := pythonMLURL() + "/trade/execute"
	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("POST", url, &byteReader{data: body})
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()
	body2, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body2)
}

// TradingPositionsSync proxies GET /trading/positions/sync → Python GET /positions/sync
func TradingPositionsSync(c *gin.Context) {
	url := pythonMLURL() + "/positions/sync"
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "ML service unavailable: " + err.Error()})
		return
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read response"})
		return
	}
	c.Data(resp.StatusCode, "application/json", body)
}

// MLMarketSignals proxies GET /ml/market-signals → Python GET /market-signals
func MLMarketSignals(c *gin.Context) {
	url := pythonMLURL() + "/market-signals"
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "ML service unavailable: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read ML response"})
		return
	}
	c.Data(resp.StatusCode, "application/json", body)
}

