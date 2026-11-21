package logging

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Level represents Log4j-style logging severity.
type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

func parseLevel(s string) Level {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "DEBUG":
		return LevelDebug
	case "WARN":
		return LevelWarn
	case "ERROR":
		return LevelError
	default:
		return LevelInfo
	}
}

func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	default:
		return "INFO"
	}
}

// DualLogger writes Jenkins-style stage entries to status.log and Log4j-style structured logs to processing.log.
type DualLogger struct {
	mu             sync.Mutex
	minLevel       Level
	statusFile     *os.File
	processingFile *os.File
	consoleOut     io.Writer
	consoleErr     io.Writer
}

// NewDualLogger initializes the log directory and opens both status and processing log files.
func NewDualLogger(logDir, statusFilename, processingFilename, levelStr string, console bool) (*DualLogger, error) {
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, fmt.Errorf("create log dir %q: %w", logDir, err)
	}
	statusPath := filepath.Join(logDir, statusFilename)
	procPath := filepath.Join(logDir, processingFilename)

	sf, err := os.OpenFile(statusPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("open status log %q: %w", statusPath, err)
	}
	pf, err := os.OpenFile(procPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		_ = sf.Close()
		return nil, fmt.Errorf("open processing log %q: %w", procPath, err)
	}

	var outW, errW io.Writer
	if console {
		outW = os.Stdout
		errW = os.Stderr
	}

	return &DualLogger{
		minLevel:       parseLevel(levelStr),
		statusFile:     sf,
		processingFile: pf,
		consoleOut:     outW,
		consoleErr:     errW,
	}, nil
}

// Status records a Jenkins-style pipeline stage transition to status.log and mirrors it to processing.log.
func (l *DualLogger) Status(stage, status, format string, args ...any) {
	if l == nil {
		return
	}
	msg := fmt.Sprintf(format, args...)
	ts := time.Now().UTC().Format(time.RFC3339)
	statusLine := fmt.Sprintf("[%s] [%s] [%s] %s\n", ts, strings.ToUpper(stage), strings.ToUpper(status), msg)
	procLine := fmt.Sprintf("[%s] [INFO] [stage.%s] (%s) %s\n", ts, strings.ToLower(stage), strings.ToUpper(status), msg)

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.statusFile != nil {
		_, _ = io.WriteString(l.statusFile, statusLine)
	}
	if l.processingFile != nil {
		_, _ = io.WriteString(l.processingFile, procLine)
	}
	if l.consoleOut != nil {
		_, _ = io.WriteString(l.consoleOut, statusLine)
	}
}

func (l *DualLogger) log(lvl Level, component, format string, args ...any) {
	if l == nil || lvl < l.minLevel {
		return
	}
	msg := fmt.Sprintf(format, args...)
	ts := time.Now().UTC().Format(time.RFC3339)
	line := fmt.Sprintf("[%s] [%s] [%s] %s\n", ts, lvl.String(), component, msg)

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.processingFile != nil {
		_, _ = io.WriteString(l.processingFile, line)
	}
	if lvl >= LevelWarn && l.consoleErr != nil {
		_, _ = io.WriteString(l.consoleErr, line)
	} else if l.consoleOut != nil {
		_, _ = io.WriteString(l.consoleOut, line)
	}
}

// Debug logs a DEBUG message to processing.log.
func (l *DualLogger) Debug(component, format string, args ...any) {
	l.log(LevelDebug, component, format, args...)
}

// Info logs an INFO message to processing.log.
func (l *DualLogger) Info(component, format string, args ...any) {
	l.log(LevelInfo, component, format, args...)
}

// Warn logs a WARN message to processing.log.
func (l *DualLogger) Warn(component, format string, args ...any) {
	l.log(LevelWarn, component, format, args...)
}

// Error logs an ERROR message to processing.log.
func (l *DualLogger) Error(component, format string, args ...any) {
	l.log(LevelError, component, format, args...)
}

// Close flushes and closes both log files.
func (l *DualLogger) Close() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	var firstErr error
	if l.statusFile != nil {
		if err := l.statusFile.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		l.statusFile = nil
	}
	if l.processingFile != nil {
		if err := l.processingFile.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		l.processingFile = nil
	}
	return firstErr
}
