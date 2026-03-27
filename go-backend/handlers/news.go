package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type newsResponse struct {
	Articles []struct {
		Title       string `json:"title"`
		Description string `json:"description"`
	} `json:"articles"`
}

// keyword → weight mapping
// Same logic as your Python news_client.py
var shockKeywords = map[string]float64{
	"hormuz":                1.0,
	"strait of hormuz":      1.0,
	"oil supply disruption": 0.9,
	"tanker attack":         0.85,
	"iran strikes":          0.8,
	"iran war":              0.75,
	"middle east conflict":  0.7,
	"opec cut":              0.7,
	"energy crisis":         0.6,
	"fuel rationing":        0.75,
	"oil price surge":       0.6,
	"crude spike":           0.6,
	"oil sanctions":         0.65,
	"brent crude":           0.3,
	"oil price":             0.3,
}

func fetchHeadlineScore() (float64, error) {
	apiKey := os.Getenv("GNEWS_API_KEY")
	if apiKey == "" {
		return 0, fmt.Errorf("GNEWS_API_KEY not set")
	}

	url := fmt.Sprintf(
		"https://gnews.io/api/v4/search?q=oil+iran+energy&lang=en&max=10&token=%s",
		apiKey,
	)

	resp, err := http.Get(url)
	if err != nil {
		return 0, fmt.Errorf("GNewsAPI request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("reading GNewsAPI response: %w", err)
	}
	fmt.Printf("GNews status: %d\n", resp.StatusCode)
	fmt.Printf("GNews raw: %s\n", string(body[:min(300, len(body))]))
	var result newsResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("parsing GNewsAPI json: %w", err)
	}

	return scoreArticles(result.Articles), nil
}

func scoreArticles(articles []struct {
	Title       string `json:"title"`
	Description string `json:"description"`
}) float64 {
	maxScore := 0.0

	for _, article := range articles {
		fmt.Println("Article:", article.Title)
		combined := strings.ToLower(article.Title + " " + article.Description)
		articleScore := 0.0

		for keyword, weight := range shockKeywords {
			if strings.Contains(combined, keyword) {
				if weight > articleScore {
					articleScore = weight
				}
				fmt.Printf("Matched [%.2f]: %s → %q\n", weight, keyword, article.Title)
			}
		}

		if articleScore > maxScore {
			maxScore = articleScore
		}
	}

	return maxScore
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
