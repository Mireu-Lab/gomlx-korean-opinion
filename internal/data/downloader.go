package data

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
)

type downloadSpec struct {
	name string
	url  string
	path string
}

// EnsureDatasets downloads missing NSMC and KLUE-YNAT files into cfg.Paths.DataDir.
func EnsureDatasets(cfg *config.Config, log *logging.DualLogger) error {
	if err := os.MkdirAll(cfg.Paths.DataDir, 0755); err != nil {
		return fmt.Errorf("create data dir %q: %w", cfg.Paths.DataDir, err)
	}

	specs := []downloadSpec{
		{name: "NSMC Train", url: cfg.Datasets.NSMCTrainURL, path: cfg.NSMCTrainPath()},
		{name: "NSMC Test", url: cfg.Datasets.NSMCTestURL, path: cfg.NSMCTestPath()},
		{name: "KLUE-YNAT Train", url: cfg.Datasets.YNATTrainURL, path: cfg.YNATTrainPath()},
		{name: "KLUE-YNAT Validation", url: cfg.Datasets.YNATValURL, path: cfg.YNATValPath()},
	}

	client := &http.Client{
		Timeout: time.Duration(cfg.Runtime.HTTPTimeoutSec) * time.Second,
	}

	for _, sp := range specs {
		if st, err := os.Stat(sp.path); err == nil && st.Size() > 0 {
			log.Info("downloader", "%s already cached at %s (%d bytes)", sp.name, sp.path, st.Size())
			continue
		}
		log.Info("downloader", "downloading %s from %s -> %s", sp.name, sp.url, sp.path)
		if err := downloadWithRetry(client, sp.url, sp.path, cfg.Runtime.HTTPRetries, log); err != nil {
			return fmt.Errorf("download %s: %w", sp.name, err)
		}
	}
	return nil
}

func downloadWithRetry(client *http.Client, url, destPath string, maxRetries int, log *logging.DualLogger) error {
	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		if err := downloadFileOnce(client, url, destPath); err != nil {
			lastErr = err
			log.Warn("downloader", "attempt %d/%d failed for %s: %v", attempt, maxRetries, url, err)
			if attempt < maxRetries {
				time.Sleep(time.Duration(attempt*attempt) * time.Second)
			}
			continue
		}
		return nil
	}
	return lastErr
}

func downloadFileOnce(client *http.Client, url, destPath string) error {
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "gomlx-korean-opinion/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("unexpected HTTP status %s", resp.Status)
	}

	tmpPath := destPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	written, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmpPath)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmpPath)
		return closeErr
	}
	if written == 0 {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("downloaded empty file from %s", url)
	}
	if err := os.Rename(tmpPath, destPath); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return nil
}
