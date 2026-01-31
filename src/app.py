#!/usr/bin/env python3
"""
src/app.py

Gradio-Oberfläche für die hybride Suche (semantisch + keyword) mit modernem Design.
Die hybride Suche kombiniert:
- Semantische Suche (FAISS) — liefert beliebig viele Treffer (config: k)
- Keyword-Suche (Pandas simple matching) — Top-5 Treffer
Die finalen Treffer sind eine deduplizierte Kombination (semantische Treffer zuerst, dann keyword).
"""

from __future__ import annotations

import os
from typing import List, Optional

import gradio as gr
import pandas as pd

from .search.engine import ProjectSearchEngine
from .utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Modernes CSS-Styling mit MAXIMALEM Kontrast
CUSTOM_CSS = """
/* ===== Globale Variablen & Theme ===== */
:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --secondary-color: #06b6d4;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --dark-bg: #0f172a;
    --card-bg: #1e293b;
    --border-color: #475569;
    --text-primary: #ffffff;
    --text-secondary: #e2e8f0;
    --text-muted: #cbd5e1;
    --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    --border-radius: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== Body & Container ===== */
body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* ===== KRITISCH: Spezifische Gradio-Selektoren ===== */

/* Labels und Spans - WEISS */
span.svelte-g2oxp3,
span[data-testid="block-info"],
.svelte-g2oxp3 {
    color: #ffffff !important;
    font-weight: 600 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
}

/* Radio Button Spans - WEISS */
label.svelte-1bx8sav span.svelte-1bx8sav,
.svelte-1bx8sav span {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Slider Labels und Werte */
label.svelte-1kajgn1,
.svelte-1kajgn1 label,
span.min_value,
span.max_value {
    color: #ffffff !important;
}

/* Number Input im Slider - DUNKEL AUF HELL */
input.svelte-1kajgn1[type="number"] {
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 2px solid var(--border-color) !important;
    font-weight: 600 !important;
}

/* Textareas - Unterscheiden zwischen editierbar und readonly */
textarea.svelte-1ae7ssi:not([disabled]) {
    background: #f8fafc !important;
    color: #1e293b !important;
}

textarea.svelte-1ae7ssi[disabled] {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #ffffff !important;
}

/* ===== Header Styling ===== */
.markdown-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: 2.5rem 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    text-align: center;
}

.markdown-header h2 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.5rem 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.markdown-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin: 0 !important;
}

/* ===== Cards & Blocks ===== */
.block {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: var(--transition) !important;
}

.block:hover {
    box-shadow: var(--shadow-lg) !important;
    transform: translateY(-2px);
}

/* ===== Input Fields ===== */
.input-container label,
label.svelte-1gfkn6j,
label,
.gr-box label,
.gr-form label,
fieldset legend,
.wrap.svelte-1gfkn6j > label,
span.svelte-1gfkn6j {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
    background: transparent !important;
}

input[type="text"],
textarea {
    background: #f8fafc !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
}

input[type="text"]:focus,
textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    outline: none !important;
}

input[type="text"]::placeholder,
textarea::placeholder {
    color: #64748b !important;
}

/* ===== Buttons ===== */
.primary-button,
button[type="submit"] {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-md) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.primary-button:hover,
button[type="submit"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg) !important;
}

.primary-button:active,
button[type="submit"]:active {
    transform: translateY(0);
}

/* ===== Radio Buttons - MAXIMALER KONTRAST ===== */
.radio-group,
fieldset.svelte-1gfkn6j,
fieldset {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

.radio-group label,
fieldset label,
.svelte-1gfkn6j label,
label[for] {
    color: #ffffff !important;
    padding: 0.75rem 1rem !important;
    border-radius: 6px !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
    font-weight: 600 !important;
    background: rgba(30, 41, 59, 0.8) !important;
    margin: 0.25rem !important;
    border: 2px solid var(--border-color) !important;
    display: inline-block !important;
}

.radio-group input[type="radio"]:checked + label,
input[type="radio"]:checked + label,
input[type="radio"]:checked ~ label {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
}

.radio-group label:hover,
fieldset label:hover {
    background: rgba(99, 102, 241, 0.3) !important;
    color: #ffffff !important;
    border-color: var(--primary-color) !important;
}

/* Fieldset Legend (z.B. "Suchmodus") */
fieldset legend,
legend {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.5rem 1rem !important;
    background: rgba(99, 102, 241, 0.3) !important;
    border-radius: 6px !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
}

/* ===== Slider - SCHWARZER TEXT AUF HELLEM GRUND ===== */
.slider-container {
    padding: 1rem !important;
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 8px !important;
}

.slider-container label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Slider-Wert-Anzeige (Number Input) */
input[type="number"] {
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: var(--border-color);
    outline: none;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--primary-color);
    cursor: pointer;
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}

input[type="range"]::-webkit-slider-thumb:hover {
    background: var(--primary-hover);
    transform: scale(1.2);
}

input[type="range"]::-moz-range-thumb {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--primary-color);
    cursor: pointer;
    border: none;
    box-shadow: var(--shadow-md);
}

/* ===== DataTable - MIT SCROLLBAR UND KONTRAST ===== */
.dataframe,
table,
.gr-table,
div[class*="table-wrap"] {
    background: var(--dark-bg) !important;
    border-radius: var(--border-radius) !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
    max-height: 600px !important;
    box-shadow: var(--shadow-md) !important;
    display: block !important;
}

.dataframe thead,
table thead,
.gr-table thead {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
}

.dataframe thead th,
table thead th,
.gr-table thead th,
th {
    color: #ffffff !important;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    font-weight: 700 !important;
    padding: 1rem !important;
    text-align: left !important;
    border-bottom: 2px solid var(--border-color) !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
}

.dataframe tbody tr,
table tbody tr,
.gr-table tbody tr {
    border-bottom: 1px solid var(--border-color) !important;
    transition: var(--transition) !important;
    background: rgba(15, 23, 42, 0.8) !important;
}

.dataframe tbody tr:hover,
table tbody tr:hover,
.gr-table tbody tr:hover {
    background: rgba(99, 102, 241, 0.2) !important;
}

.dataframe tbody td,
table tbody td,
.gr-table tbody td,
td {
    color: #ffffff !important;
    background: transparent !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
}

/* Gradio spezifische DataFrame-Klassen */
.gr-dataframe {
    overflow: auto !important;
    max-height: 600px !important;
}

.gr-dataframe table {
    background: rgba(15, 23, 42, 0.9) !important;
}

.gr-dataframe th {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    color: #ffffff !important;
    position: sticky !important;
    top: 0 !important;
}

.gr-dataframe td {
    color: #ffffff !important;
    background: rgba(15, 23, 42, 0.6) !important;
}

/* ===== Textbox (LLM Answer) - WEISS AUF DUNKEL MIT SCROLLBAR ===== */
textarea[readonly],
.output-textbox {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    color: #ffffff !important;
    padding: 1.5rem !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: var(--shadow-md) !important;
    overflow-y: auto !important;
    max-height: 600px !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}

/* Alle Textboxen die interaktiv sind: dunkel auf hell */
textarea:not([readonly]) {
    background: #f8fafc !important;
    color: #1e293b !important;
}

/* Spezifische Gradio Textbox */
.gr-textbox textarea[readonly] {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #ffffff !important;
}

.gr-textbox textarea:not([readonly]) {
    background: #f8fafc !important;
    color: #1e293b !important;
}

/* ===== Tabs ===== */
.tabs,
.tab-nav {
    border-bottom: 2px solid var(--border-color) !important;
    margin-bottom: 1.5rem !important;
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: var(--border-radius) var(--border-radius) 0 0 !important;
}

.tab-nav button,
button.svelte-1b6s6s {
    color: var(--text-secondary) !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 1rem 1.5rem !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
    background: transparent !important;
    font-size: 1rem !important;
}

.tab-nav button:hover {
    color: var(--text-primary) !important;
    background: rgba(99, 102, 241, 0.15) !important;
}

.tab-nav button.selected,
button.selected.svelte-1b6s6s {
    color: var(--primary-color) !important;
    border-bottom-color: var(--primary-color) !important;
    background: rgba(99, 102, 241, 0.1) !important;
    font-weight: 700 !important;
}

/* ===== Loading Animation ===== */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-radius: 50%;
    border-top-color: var(--primary-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ===== Info Cards ===== */
.info-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
    border-left: 4px solid var(--primary-color);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-md);
}

.info-card-title {
    color: var(--primary-color);
    font-weight: 700;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.info-card-content {
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.7;
    font-weight: 400;
}

/* ===== Verbesserte Lesbarkeit für alle Texte ===== */
p, span, div, .markdown {
    color: var(--text-primary) !important;
}

/* Footer und sekundäre Texte */
.footer-text,
.secondary-text {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* Beispiele Section */
.gr-samples-table,
.gr-sample-textbox {
    color: var(--text-primary) !important;
    background: rgba(30, 41, 59, 0.6) !important;
}

/* Headers in Markdown */
.markdown h1,
.markdown h2,
.markdown h3,
.markdown h4 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* Gradio-spezifische Klassen */
.gr-padded,
.gr-box,
.gr-form {
    color: var(--text-primary) !important;
}

/* ===== Beispiele Section Styling ===== */
.examples-header h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
    margin: 1.5rem 0 1rem 0 !important;
    padding: 0.5rem !important;
    background: rgba(99, 102, 241, 0.1) !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--primary-color) !important;
}

/* ===== Custom Examples Wrapper ===== */
.examples-wrapper,
.custom-examples {
    background: rgba(15, 23, 42, 0.95) !important;
    padding: 1.5rem !important;
    border-radius: var(--border-radius) !important;
    border: 2px solid var(--border-color) !important;
}

.custom-examples * {
    color: #ffffff !important;
    background: transparent !important;
}

.custom-examples table {
    background: rgba(30, 41, 59, 0.8) !important;
}

.custom-examples td,
.custom-examples th {
    color: #ffffff !important;
    border-color: var(--border-color) !important;
}

.custom-examples button {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #ffffff !important;
    border: 2px solid var(--border-color) !important;
}

.custom-examples button:hover {
    background: rgba(99, 102, 241, 0.4) !important;
    border-color: var(--primary-color) !important;
}
.footer-section {
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: var(--border-radius) !important;
    padding: 1rem !important;
    margin-top: 2rem !important;
}

.footer-section p {
    margin: 0.5rem 0 !important;
}

.footer-section strong {
    font-weight: 700 !important;
}

/* ===== Accordion - EXAKTE SELEKTOREN ===== */
button.label-wrap.svelte-1w6vloh {
    background: rgba(99, 102, 241, 0.3) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    padding: 1rem !important;
    border-radius: 8px !important;
}

button.label-wrap.svelte-1w6vloh span.svelte-1w6vloh {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Accordion Content - WEISSER TEXT */
.prose.svelte-lag733 {
    color: #ffffff !important;
}

.prose.svelte-lag733 *,
.prose.svelte-lag733 h3,
.prose.svelte-lag733 p,
.prose.svelte-lag733 ul,
.prose.svelte-lag733 li,
.prose.svelte-lag733 strong {
    color: #ffffff !important;
}

/* Markdown Content */
span.md.svelte-7ddecg.prose,
.md.svelte-7ddecg *,
.md.svelte-7ddecg h3,
.md.svelte-7ddecg p,
.md.svelte-7ddecg ul,
.md.svelte-7ddecg li {
    color: #ffffff !important;
}

/* Bessere Kontraste für interaktive Elemente */
button, input, select, textarea {
    color: var(--text-primary) !important;
}

/* Gradio Examples Table */
.gr-samples-table td {
    color: var(--text-primary) !important;
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid var(--border-color) !important;
}

.gr-samples-table th {
    color: var(--text-primary) !important;
    background: rgba(99, 102, 241, 0.2) !important;
    font-weight: 600 !important;
}
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }

    .markdown-header {
        padding: 1.5rem 1rem;
    }

    .markdown-header h2 {
        font-size: 1.8rem !important;
    }

    .block {
        padding: 1rem !important;
    }
}

/* ===== Scrollbar Styling ===== */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-color);
}

/* ===== Animations ===== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.block {
    animation: fadeIn 0.5s ease-out;
}

/* ===== Status Badges ===== */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-badge.success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success-color);
}

.status-badge.warning {
    background: rgba(245, 158, 11, 0.2);
    color: var(--warning-color);
}

.status-badge.info {
    background: rgba(6, 182, 212, 0.2);
    color: var(--secondary-color);
}

/* ===== Spaltenbreiten für DataTable ===== */
.gr-dataframe td:nth-child(3),  /* 3. Spalte = Thema */
.gr-dataframe th:nth-child(3) {
    min-width: 200px !important;
    max-width: 400px !important;
    width: 300px !important;
}

.gr-dataframe td:nth-child(1),  /* FKZ */
.gr-dataframe th:nth-child(1) {
    min-width: 100px !important;
    width: 120px !important;
}

.gr-dataframe td:nth-child(2),  /* Zuwendungsempfänger */
.gr-dataframe th:nth-child(2) {
    min-width: 100px !important;
    max-width: 180px !important;
}

.gr-dataframe td:nth-child(4),  /* Bundesland */
.gr-dataframe th:nth-child(4) {
    min-width: 80px !important;
    max-width: 120px !important;
    width: 100px !important;
}

.gr-dataframe td:nth-child(5),  /* Laufzeit */
.gr-dataframe th:nth-child(5) {
    min-width: 100px !important;
    width: 120px !important;
}

.gr-dataframe td:nth-child(6),  /* Fördersumme */
.gr-dataframe th:nth-child(6) {
    min-width: 60px !important;
    width: 100px !important;
}

.gr-dataframe td:nth-child(7),  /* Score */
.gr-dataframe th:nth-child(7) {
    min-width: 80px !important;
    width: 100px !important;
}

/* Tabelle darf horizontal scrollen */
.dataframe,
table,
.gr-table,
div[class*="table-wrap"] {
    overflow-x: auto !important;
    white-space: nowrap !important;
}

/* Begrenze die tatsächliche Tabellenbreite */
.dataframe table,
.gr-table table,
table {
    max-width: 1600px !important;  /* Maximale Gesamtbreite der Tabelle */
    width: auto !important;
}

/* ===== Examples Table - Volle Breite ===== */
.gradio-container .gr-samples-table,
.gradio-container div[data-testid*="sample"],
.gradio-container .gr-sample-textbox {
    width: 100% !important;
    max-width: 100% !important;
}

div.gr-form.gr-box > div {
    width: 100% !important;
}

/* Spezifische Gradio Examples Wrapper */
div[id*="example"] {
    width: 100% !important;
}

div[id*="example"] table {
    width: 100% !important;
    table-layout: auto !important;
}

div[id*="example"] td,
div[id*="example"] th {
    padding: 0.75rem 1rem !important;
}

/* Examples Container */
.gr-sample-textbox,
button[id*="example"] {
    width: 100% !important;
    max-width: 100% !important;
}

/* ===== Dropdown - Dunkler Hintergrund ===== */
select,
.gr-dropdown,
div[data-testid="dropdown"],
.svelte-1gfkn6j select {
    background: var(--card-bg) !important;
    color: #ffffff !important;
    border: 2px solid var(--border-color) !important;
}

/* Dropdown-Optionen */
select option {
    background: var(--card-bg) !important;
    color: #ffffff !important;
    padding: 0.5rem !important;
}

select option:hover,
select option:focus {
    background: rgba(99, 102, 241, 0.3) !important;
}

/* Dropdown-Container */
.wrap.svelte-1gfkn6j.svelte-1gfkn6j,
div.wrap.svelte-1gfkn6j {
    background: transparent !important;
}

/* Dropdown Hover/Focus States */
select:hover,
select:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}
"""


def get_project_details(fkz: str, engine: ProjectSearchEngine) -> str:
    """Liefert formatierte Detail-Informationen zu einem Projekt.

    Args:
        fkz: Förderkennzeichen des Projekts
        engine: ProjectSearchEngine-Instanz

    Returns:
        str: Formatierte Projekt-Details als Markdown
    """
    if engine.df is None or not fkz:
        return "Keine Projektdaten verfügbar."

    # Suche Projekt anhand FKZ
    fkz_col = '="FKZ"'
    if fkz_col not in engine.df.columns:
        return f"FKZ-Spalte nicht gefunden. Verfügbare Spalten: {', '.join(engine.df.columns)}"

    # Bereinige FKZ (entferne Formatierung)
    search_fkz = fkz.strip()

    # Finde Projekt
    matches = engine.df[engine.df[fkz_col].astype(str).str.contains(search_fkz, case=False, na=False)]

    if matches.empty:
        return f"❌ Kein Projekt mit FKZ '{fkz}' gefunden."

    project = matches.iloc[0]

    # Erstelle detaillierte Ansicht mit kleinerer Schrift
    details = f'<div style="font-size: 0.9rem;">\n\n# 📋 Projekt-Details: {fkz}\n\n'

    # Wichtige Felder zuerst
    important_fields = [
        ('="FKZ"', "🔑 Förderkennzeichen"),
        ('="Zuwendungsempfänger"', "🏢 Zuwendungsempfänger"),
        ('="Thema"', "📝 Thema"),
        ('="Fördersumme in EUR"', "💰 Fördersumme"),
        ("__laufzeit", "📅 Laufzeit"),
        ('="Laufzeit von"', "📅 Start"),
        ('="Laufzeit bis"', "📅 Ende"),
    ]

    for col, label in important_fields:
        if col in project.index:
            val = project[col]
            if pd.notna(val) and str(val).strip() and str(val) != "nan":
                if col == '="Fördersumme in EUR"':
                    try:
                        val = f"{float(val):,.2f} €"
                    except (ValueError, TypeError):
                        pass
                details += f"**{label}:** {val}\n"

    details += "---\n## 📍 Weitere Informationen\n"

    # Alle anderen Felder
    other_fields = [
        ('="Ausführende Stelle"', "🏛️ Ausführende Stelle"),
        ('="Stadt/Gemeinde"', "🏙️ Stadt/Gemeinde"),
        ('="Postleitzahl"', "📮 PLZ"),
        ('="Bundesland"', "🗺️ Bundesland"),
        ('="Verbundprojekt"', "🤝 Verbundprojekt"),
        ('="Förderprofil"', "📊 Förderprofil"),
        ('="Klartext Leistungsplansystematik"', "📚 Leistungsplansystematik"),
        ('="Projektträger"', "👥 Projektträger"),
    ]

    for col, label in other_fields:
        if col in project.index:
            val = project[col]
            if pd.notna(val) and str(val).strip() and str(val) != "nan":
                details += f"• **{label}:** {val}\n"

    details += "</div>"

    return details


# ÄNDERUNG 2: Modifiziere die LLM-Antwort-Generierung (in answer_with_context)
# Füge in src/search/engine.py nach Zeile 252 ein:
def make_fkz_clickable(answer: str) -> str:
    """Macht FKZ in der Antwort zu klickbaren Buttons.

    Args:
        answer: LLM-Antwort mit FKZ-Referenzen

    Returns:
        str: Antwort mit HTML-Buttons für FKZ
    """
    import re

    # Finde alle FKZ-Muster (z.B. 13BDB60030, 01AB1234)
    pattern = r"\*\*FKZ:\s*([A-Z0-9]+)\*\*"

    def replace_fkz(match):
        fkz = match.group(1)
        # Erstelle anklickbaren Button-Style Text
        return f"**FKZ: [{fkz}](#{fkz})**"

    return re.sub(pattern, replace_fkz, answer)


def extract_fkz_from_text(text: str) -> list:
    """Extrahiert alle FKZ aus einem Text.

    Args:
        text: Text mit FKZ-Referenzen (z.B. LLM-Antwort)

    Returns:
        list: Liste von gefundenen FKZ (ohne Duplikate)

    Example:
        >>> extract_fkz_from_text("Projekt FKZ: 13BDB60030 und FKZ 01AB1234")
        ['13BDB60030', '01AB1234']
    """
    import re

    # Muster: FKZ: 13BDB60030 oder FKZ 01AB1234 (mindestens 8 Zeichen)
    pattern = r"FKZ[:\s]+([A-Z0-9]{8,})"
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Duplikate entfernen, Reihenfolge beibehalten
    seen = set()
    unique_fkz = []
    for fkz in matches:
        if fkz not in seen:
            seen.add(fkz)
            unique_fkz.append(fkz)

    return unique_fkz


def keyword_search(df: pd.DataFrame, query: str, top_n: int = 5, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Einfache keyword-/substring-Suche über ausgewählte Textspalten.

    Bewertet Ergebnisse nach Anzahl der Treffer (Count) und gibt top_n zurück.

    Args:
        df: DataFrame mit Projektdaten.
        query: Nutzeranfrage (String).
        top_n: Anzahl der zurückzugebenden Treffer. Defaults to 5.
        text_columns: Liste von Spalten, die durchsucht werden.
            Wenn None, werden gängige Spalten genutzt.

    Returns:
        pd.DataFrame: DataFrame mit top_n Treffern (inkl. Spalte '__kw_score').

    Example:
        >>> results = keyword_search(df, "KI Bayern", top_n=5)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    qc = query.strip().lower()
    if not qc:
        return pd.DataFrame()

    # Auswahl der Spalten
    text_columns = text_columns or [
        c
        for c in (
            "Zuwendungsempfänger",
            "Zuwendungsempf\u00e4nger",
            "Thema",
            "Klartext Leistungsplansystematik",
            "Ausf\u00fchrende Stelle",
        )
        if c in df.columns
    ]

    scores = []
    for idx, row in df.iterrows():
        score = 0
        for col in text_columns:
            val = str(row.get(col, "")).lower()
            if not val:
                continue
            # weight: occurrences of whole query and individual token matches
            if qc in val:
                score += 10
            for token in qc.split():
                if token in val:
                    score += 1
        if score > 0:
            scores.append((idx, score))

    if not scores:
        return pd.DataFrame()

    # Top-N nach Score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    indices = [idx for idx, _ in scores]
    result = df.loc[indices].copy()
    result["__kw_score"] = [s for _, s in scores]
    return result


def hybrid_rank(sem_df: pd.DataFrame, kw_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Kombiniert semantische Ergebnisse (sem_df) und keyword Ergebnisse (kw_df).

    Priorisiert semantische Treffer und hängt keyword-Treffer an, ohne Duplikate.

    Args:
        sem_df: DataFrame der semantischen Treffer (mit '__score' Spalte).
        kw_df: DataFrame der keyword Treffer (mit '__kw_score' Spalte).
        k: Gewünschte Gesamtanzahl der finalen Treffer.

    Returns:
        pd.DataFrame: DataFrame mit maximal k Treffern.
    """
    if sem_df is None:
        sem_df = pd.DataFrame()
    if kw_df is None:
        kw_df = pd.DataFrame()

    # Beginne mit semantischen Treffern (nach Score sortiert)
    sem_df_sorted = sem_df.sort_values("__score", ascending=False) if "__score" in sem_df.columns else sem_df
    combined = []
    added_ids = set()

    # Add semantic first
    for idx, row in sem_df_sorted.iterrows():
        if len(combined) >= k:
            break
        if idx in added_ids:
            continue
        combined.append(row)
        added_ids.add(idx)

    # Then keyword results
    kw_df_sorted = kw_df.sort_values("__kw_score", ascending=False) if "__kw_score" in kw_df.columns else kw_df
    for idx, row in kw_df_sorted.iterrows():
        if len(combined) >= k:
            break
        if idx in added_ids:
            continue
        combined.append(row)
        added_ids.add(idx)

    if not combined:
        return pd.DataFrame()

    result = pd.DataFrame(combined)
    return result


def build_ui(engine: ProjectSearchEngine) -> gr.Blocks:
    """Baut die Gradio-Benutzeroberfläche mit FKZ-Detail-Ansicht."""

    with gr.Blocks(
        title="🧠 RAG Förderkatalog — Intelligente Projektsuche",
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=("Inter", "system-ui", "sans-serif"),
        ),
    ) as demo:

        # Header
        gr.Markdown(
            """
            ## 🧠 RAG Förderkatalog
            **Intelligente Suche in deutschen Forschungsförderprojekten**
            """,
            elem_classes="markdown-header",
        )

        groq_key_preset = os.getenv("GROQ_API_KEY")

        if not groq_key_preset:
            with gr.Row():
                gr.Markdown("""
                    <div class="info-card">
                        <div class="info-card-title">🔑 API-Konfiguration erforderlich</div>
                        <div class="info-card-content">
                            Für KI-Antworten benötigen Sie einen GROQ API-Key.
                            <a href="https://console.groq.com/" target="_blank" style="color: #6366f1; font-weight: 600;">
                            Hier kostenlos registrieren</a>.
                        </div>
                    </div>
                    """)

            with gr.Row():
                api_key_input = gr.Textbox(
                    label="🔑 GROQ API Key",
                    placeholder="gsk_...",
                    type="password",
                    elem_classes="input-container",
                )
                api_status = gr.Markdown("⚠️ Kein API-Key gesetzt", elem_classes="footer-text")

        # Info Card
        # Info Card - mit dynamischen Projektzahlen
        with gr.Row():
            total_projects = len(engine.df) if engine.df is not None else 0
            indexed_projects = engine.faiss.index.ntotal if engine.faiss.index is not None else 0

            gr.Markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">💡 Über diese Anwendung</div>
                    <div class="info-card-content">
                        Diese KI-gestützte Suchmaschine durchsucht über {total_projects:,} Förderprojekte des Bundes
                        (davon indiziert: {indexed_projects:,})
                        mit semantischer Vektorsuche und kontextbasierter KI-Antwortgenerierung.
                        Wählen Sie zwischen <strong>Semantischer</strong>, <strong>Keyword</strong> oder
                        <strong>Hybrider Suche</strong> für optimale Ergebnisse.
                    </div>
                </div>
                """)

        # Search Controls
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="🔍 Suchanfrage",
                    placeholder="z. B. 'Künstliche Intelligenz Projekte Bayern 2020-2025'",
                    lines=2,
                    elem_classes="input-container",
                )

        with gr.Row():
            with gr.Column(scale=2):
                mode = gr.Radio(
                    choices=["hybrid", "semantic", "keyword"],
                    value="hybrid",
                    label="⚙️ Suchmodus",
                    elem_classes="radio-group",
                )
            with gr.Column(scale=2):
                k_slider = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=20,
                    step=5,
                    label="📊 Anzahl Treffer (k)",
                    elem_classes="slider-container",
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🚀 Suchen", variant="primary", elem_classes="primary-button")

        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("📋 Suchergebnisse"):
                result_table = gr.Dataframe(
                    headers=None,
                    interactive=False,
                    label="Gefundene Projekte",
                    wrap=True,
                    elem_classes="dataframe",
                )

                with gr.Accordion("📈 Statistiken", open=False):
                    stats_output = gr.Markdown()

            with gr.Tab("🤖 KI-Analyse"):
                llm_answer = gr.Textbox(
                    label="Kontextbasierte KI-Antwort",
                    interactive=False,
                    lines=15,
                    max_lines=15,
                    elem_classes="output-textbox",
                )

                # FKZ-Dropdown für Projekt-Auswahl
                gr.Markdown("### 🔑 In der Antwort erwähnte Projekte")
                fkz_radio = gr.Radio(
                    label="Wählen Sie ein Projekt für Details aus",
                    choices=[],
                    interactive=True,
                    visible=False,
                    elem_classes="radio-group",
                )

                # Detail-Ansicht
                detail_output = gr.Markdown(value="", visible=False, elem_classes="output-textbox")

        # Examples
        gr.Markdown("### 💡 Beispielsuchen", elem_classes="examples-header")

        with gr.Row():
            gr.Examples(
                examples=[
                    ["Künstliche Intelligenz Hochschule Bayern", "hybrid", 20],
                    ["Wasserstoff Energie NRW 2020-2025", "semantic", 15],
                    ["Quantencomputing Forschung", "semantic", 25],
                    ["Klimawandel Digitalisierung", "hybrid", 30],
                    ["KI zur Regelung von Biogasanlagen", "hybrid", 20],
                ],
                inputs=[query_input, mode, k_slider],
                label="Klicken Sie auf ein Beispiel zum Ausprobieren",
                examples_per_page=5,
            )

        # Search Function
        def on_search(query: str, mode_choice: str, k: int, api_key: str = ""):
            """Führt die Suche durch und generiert Ergebnisse."""

            # Setze API-Key wenn vom User bereitgestellt
            if api_key and api_key.strip():
                os.environ["GROQ_API_KEY"] = api_key.strip()

            logger.info("Suchanfrage: mode=%s, k=%s, query=%s", mode_choice, k, query)

            if not query or not query.strip():
                return (
                    pd.DataFrame(),
                    "⚠️ Bitte geben Sie eine Suchanfrage ein.",
                    "",
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False, value=""),
                )

            if engine.df is None:
                return (
                    pd.DataFrame(),
                    "❌ DataFrame nicht geladen.",
                    "",
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False, value=""),
                )

            sem_df = pd.DataFrame()
            kw_df = pd.DataFrame()

            try:
                if mode_choice in ("hybrid", "semantic"):
                    sem_df = engine.search(query, k=int(k))
                    logger.info("Semantische Suche: %d Treffer gefunden", len(sem_df))
                if mode_choice in ("hybrid", "keyword"):
                    kw_df = keyword_search(engine.df, query, top_n=5)
                    logger.info("Keyword-Suche: %d Treffer gefunden", len(kw_df))

                if mode_choice == "semantic":
                    final = sem_df.head(int(k)) if not sem_df.empty else pd.DataFrame()
                elif mode_choice == "keyword":
                    final = kw_df
                else:  # hybrid
                    final = hybrid_rank(sem_df, kw_df, int(k))

                if final is None or final.empty:
                    return (
                        pd.DataFrame(),
                        "🔍 Keine Treffer gefunden.",
                        "",
                        gr.update(choices=[], visible=False),
                        gr.update(visible=False, value=""),
                    )

                # Display columns
                display_cols = [
                    c
                    for c in (
                        '="FKZ"',
                        '="Zuwendungsempfänger"',
                        "Zuwendungsempf\u00e4nger",
                        '="Thema"',
                        '="Bundesland"',
                        "__laufzeit",
                        '="Fördersumme in EUR"',
                        "__score",
                        "__kw_score",
                    )
                    if c in final.columns
                ]
                display_df = final[display_cols].copy()

                # Formatierung
                if '="Fördersumme in EUR"' in display_df.columns:
                    display_df['="Fördersumme in EUR"'] = display_df['="Fördersumme in EUR"'].apply(
                        lambda x: f"{x:,.2f} €" if pd.notna(x) else "N/A"
                    )

                if "__score" in display_df.columns:
                    display_df["__score"] = display_df["__score"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

                # Statistiken
                total_results = len(final)
                total_funding = final['="Fördersumme in EUR"'].sum() if '="Fördersumme in EUR"' in final.columns else 0

                if isinstance(total_funding, str):
                    total_funding = 0

                stats_md = f"""
                ### 📊 Suchstatistiken

                - **Treffer gefunden**: {total_results}
                - **Gesamtfördersumme**: {total_funding:,.2f} €
                - **Suchmodus**: {mode_choice.upper()}
                - **Query**: _{query}_
                """

                # LLM Antwort - mit API-Key-Check
                try:
                    if not os.getenv("GROQ_API_KEY"):
                        answer = "⚠️ **API-Key fehlt**: Bitte geben Sie einen GROQ API-Key ein, um KI-Antworten zu erhalten."
                        dropdown_update = gr.update(choices=[], visible=False)
                    else:
                        answer = engine.answer_with_context(query)

                    # Extrahiere FKZ aus Antwort
                    fkz_list = extract_fkz_from_text(answer)

                    # Erstelle Dropdown-Choices mit FKZ und Empfänger
                    fkz_choices = []
                    for fkz in fkz_list:
                        # Suche Empfänger für bessere Anzeige
                        if '="FKZ"' in final.columns:
                            match = final[final['="FKZ"'].astype(str).str.contains(fkz, case=False, na=False)]
                            if not match.empty:
                                empfaenger = match.iloc[0].get('="Zuwendungsempfänger"', "N/A")
                                # Kürze Empfänger-Namen wenn zu lang
                                if len(empfaenger) > 15:
                                    empfaenger = empfaenger[:12] + "..."
                                fkz_choices.append((f"{fkz} — {empfaenger}", fkz))
                            else:
                                fkz_choices.append((fkz, fkz))

                    dropdown_update = gr.update(choices=fkz_choices, visible=len(fkz_choices) > 0, value=None)

                except Exception as e:
                    logger.exception("Fehler beim LLM-Abruf: %s", e)
                    answer = f"⚠️ LLM-Antwort konnte nicht generiert werden.\n\n**Fehler**: {str(e)}"
                    dropdown_update = gr.update(choices=[], visible=False)

                return (
                    display_df,
                    answer,
                    stats_md,
                    dropdown_update,
                    gr.update(visible=False, value=""),  # detail_output verstecken bei neuer Suche
                )

            except Exception as e:
                logger.exception("Fehler während der Suche: %s", e)
                return (
                    pd.DataFrame(),
                    f"❌ Ein Fehler ist aufgetreten: {str(e)}",
                    "",
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False, value=""),
                )

        # Detail-Anzeige-Funktion
        def show_fkz_detail(fkz):
            """Zeigt Details für ausgewähltes FKZ."""
            if not fkz:
                return gr.update(visible=False, value="")

            details = get_project_details(fkz, engine)
            return gr.update(visible=True, value=details)

        # Event Handlers
        if not groq_key_preset:
            # Mit API-Key Input
            search_btn.click(
                on_search,
                inputs=[query_input, mode, k_slider, api_key_input],
                outputs=[result_table, llm_answer, stats_output, fkz_radio, detail_output],
            )

            # API-Key Status updaten
            def update_api_status(key):
                if key and key.strip():
                    return "✅ API-Key gesetzt"
                return "⚠️ Kein API-Key gesetzt"

            api_key_input.change(update_api_status, inputs=[api_key_input], outputs=[api_status])
        else:
            # Ohne API-Key Input (für lokale Nutzung)
            search_btn.click(
                lambda q, m, k: on_search(q, m, k, ""),
                inputs=[query_input, mode, k_slider],
                outputs=[result_table, llm_answer, stats_output, fkz_radio, detail_output],
            )

        # Dropdown-Auswahl zeigt Details
        fkz_radio.change(show_fkz_detail, inputs=[fkz_radio], outputs=[detail_output])

        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; padding: 1.5rem 0;">
                <p style="color: #e2e8f0; font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">
                    🚀 Powered by <strong style="color: #6366f1;">Ollama</strong>,
                    <strong style="color: #06b6d4;">FAISS</strong>,
                    <strong style="color: #10b981;">Gradio</strong> &
                    <strong style="color: #f59e0b;">LLMClient</strong>
                </p>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    📊 Datenquelle: <a href="https://foerderportal.bund.de/foekat/jsp/SucheAction.do?actionMode=searchmask" target="_blank" style="color: #6366f1; text-decoration: none; font-weight: 600;">Förderkatalog des Bundes</a> | 🧠 RAG-basierte semantische Suche
                </p>
            </div>
            """,
            elem_classes="footer-section",
        )

    return demo


# Für den direkten Start via `python -m src.app` falls gewünscht
if __name__ == "__main__":
    setup_logging()
    logger.info("Starte Gradio-App (src.app) standalone")
    engine = ProjectSearchEngine()
    engine.load_and_clean()
    # Beim direkten Start: nur dann Embeddings erzeugen, wenn Index leer
    try:
        engine.build_embeddings_if_missing()
    except Exception:
        logger.exception("Fehler beim Erzeugen der Embeddings beim App-Start")
    demo = build_ui(engine)
    demo.launch(share=False, inbrowser=True)
