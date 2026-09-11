# Data Dictionary & Full Feature Map — FIAM-McGill Hackathon 2026

Audit method: **parquet metadata + row-group statistics only** (no data loaded). Schema dumped verbatim to
`data/raw/_schema_chars_final_with_names.parquet.txt` and `data/raw/_schema_8k_20150101_20260831_identified.parquet.txt`;
per-column null counts in `data/raw/_nulls_chars.csv` / `data/raw/_nulls_filings.csv`.

---

## 1. File format audit

| Property | `chars_final_with_names.parquet` | `8k_20150101_20260831_identified.parquet` |
|---|---|---|
| Rows | 529,082 | 373,139 |
| Columns | 198 | 94 |
| Row groups | 33 (12,182 + 31×16,384 + 8,996) | 93 (~4,000 each; last 2,399) |
| Compression | ZSTD (all columns checked) | ZSTD |
| Writer | parquet-cpp-arrow v25.0.1 | parquet-cpp-arrow v25.0.1 |
| Size | 412,599,998 B (413 MB) | 358,076,009 B (358 MB) |
| Unit | security × calendar month | retained filing observation × security |
| Keys | unique on (permno, eom) AND (gvkey, iid, eom) | unique on document_id AND (permno, filing_date, text_sha256) |
| Date coverage | `date` 2015-01-30→2026-08-28, `eom` 2015-01-31→2026-08-31 | `filing_date` 2015-01-02→2026-08-31 |
| Schema metadata | (none) | downloader_version 1.1.0; research_cleaner_version 2.0.0; row/timestamp/text/date/identifier semantic warnings (see §4.6) |

Both files: Arrow `date32[day]` for dates, UTC-timestamped columns carry timezone, list columns for items/flags, `large_string` for text.
Read pattern: use column projection + `filing_date`/`eom` filters (row groups are ~4–16k rows; ZSTD makes columnar pruning effective).

Extreme values in stats: `ret_exc_lead1m` ∈ [−1.004, +53.997] (full-loss & +5400% microcap months); `me_lag1` min ≈ 0.00076 (USD millions ⇒ ~$800 market cap). Plan for heavy right tail / microcaps.

---

## 2. TARGET VARIABLE — `ret_exc_lead1m` (chars file)

**Definition:** next-month **excess** return (decimal units, e.g. 0.01 = 1%) attached to the row whose `eom` = month t. It is the stock's excess return realized in **month t+1**.

**Rules (absolute):**
1. Use as the prediction target **without shifting again** (it is already led).
2. **Never include it as a predictor**; never use current-month `ret` / `ret_exc` as a forward target.
3. Splits (train/val/test) are assigned by the **target return month** (t+1), not by `eom` of characteristics. A Jan-2021 forecast uses the Dec-2020 characteristics row.
4. First available target month = **Feb 2015** (characteristics start Jan 2015). Do not invent Dec-2014 predictors.
5. Terminal month (`eom` 2026-08-31) has no target: **never fill with zero**.
6. Legacy-template mapping: `stock_exret` = `ret_exc_lead1m`; template `date`/`year`/`month` = target month t+1; preserve original `eom` and source `date` separately.
7. Do **not** subtract any risk-free rate a second time (RF inside `ret_exc*` is the pipeline's RF, not necessarily TB3MS).
8. Winsorized sibling `ret_exc_wins` exists (CRSP values unchanged; Compustat values clipped at period-specific 0.1%/99.9% cutoffs; **all 2026 rows are Compustat-sourced**). The competition spec names `ret_exc_lead1m` as target; winsorized variant is optional, must be documented if used.
9. Nulls: 5,957 (1.1%) → **523,125 usable labels** (missing = terminal month + unavailable outcomes; missing ≠ zero).
10. OOS R² must be computed against zero-benchmark: `1 − Σ(r−r̂)²/Σr²` (no historical-mean denominator).

**Timing logic (why the row is tradeable):** characteristics of month t (accounting + price data as of month t) become available at END of month t; positions are traded at start of month t+1; `ret_exc_lead1m` realizes at end of t+1. So the row (permno, eom=t) maps 1:1 to a trade decision at start of t+1.

Alternative target (optional): fundamental forecasting (e.g. `ebit_sale`, `niq_at`, `niq_be`) — must identify future reporting period + release date explicitly; a 3-row shift is not valid quarter identification.

---

## 3. `chars_final_with_names.parquet` — ALL 198 COLUMNS MAPPED

Types as stored. Null% = fraction of the 529,082 rows missing (from parquet statistics).

### 3.1 Keys & identifiers (5)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `id` | int64 | 0 | Upstream panel identifier; equals `permno` in this snapshot |
| `permno` | int64 | 0 | CRSP security id — **primary join key** |
| `permco` | int64 | 0 | CRSP company id (a company may have several securities) |
| `gvkey` | large_string | 0 | Compustat company id (6-char string, leading zeros) |
| `iid` | large_string | 0 | Compustat issue id — preserve as supplied |

### 3.2 Outcome & return columns (NOT features, except where noted)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `ret` | double | 0 | Current stock-month return, USD, decimal (month t — contemporaneous) |
| `ret_local` | double | 0 | Current return in local currency |
| `ret_exc` | double | 0 | Current-month excess return (month t; not a target, not next-month) |
| **`ret_exc_lead1m`** | double | 1.1 | **THE TARGET**: next-month (t+1) excess return |
| `ret_exc_wins` | double | 0 | Winsorized excess return (CRSP unchanged; Compustat clipped 0.1%/99.9%) |
| `ret_lag_dif` | int64 | 0 | Pipeline return-lag bookkeeping (int) |
| `source_crsp` | int32 | 0 | 1 = CRSP return source, 0 = Compustat (**all 2026 rows = 0**) |

### 3.3 Price, volume, market-cap (available for filtering/weighting; `prc` & `market_equity` are also among the 147)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `prc` | double | 0 | Price per share USD (also in 147 list) |
| `prc_local` | double | 0 | Price in local currency |
| `prc_high` | double | 1.1 | Month high price |
| `prc_low` | double | 1.1 | Month low price |
| `shares` | double | 0 | Shares outstanding, millions |
| `me` | double | 0 | Security market equity, USD millions |
| `me_company` | double | 0 | Company-level market equity, USD millions |
| `me_lag1` | double | 0.6 | Prior month's security ME (missing across month gaps) — range [0.00076, 5.1M] |
| `market_equity` | double | 0 | Market equity characteristic (in 147 list) |
| `adjfct` | double | 0.4 | Share adjustment factor (split basis; NOT a return/predictor) |
| `dolvol` | double | 0 | Dollar trading volume |
| `tvol` | double | 0 | Share (trading) volume |

### 3.3b Dates & alignment
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `date` | date32 | 0 | Last source return observation date of the stock-month (**can precede month-end**) |
| `eom` | date32 | 0 | Calendar month-end label — monthly alignment key |

### 3.4 Exchange / security classification (auxiliary)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `excntry` | large_string | 0 | Exchange country code |
| `exch_main` | int32 | 0 | Main-exchange flag |
| `common` | int32 | 0 | Common-stock flag |
| `primary_sec` | int32 | 0 | Primary-security flag |
| `obs_main` | int32 | 0 | Main-observation flag |
| `bidask` | int32 | 0 | Bid/ask availability flag |
| `primaryexch` | large_string | 0 | Primary exchange code |
| `conditionaltype` | large_string | 0 | Conditional security type |
| `crsp_nyse` | int32 | 0 | NYSE membership flag |
| `shrcd`/`exchcd` | int16 | 0 | CRSP share / exchange codes |
| `crsp_shrcd`/`crsp_exchcd` | int16 | 0 | CRSP source-specific codes (same, named) |
| `comp_tpci` | large_string | 94.6 | Compustat share-type code (mostly missing) |
| `comp_exchg` | int64 | 94.6 | Compustat exchange code (mostly missing) |
| `curcd` | large_string | 0 | Local currency code |
| `fx` | double | 0 | FX rate |
| `size_grp` | large_string | 0 | Size bucket from NYSE breakpoints |
| `gics` | large_string | 3.0 | GICS industry code |
| `sic` | int64 | 0.05 | SIC code |
| `naics` | int64 | 0.04 | NAICS code |
| `ff49` | int32 | 1.1 | Fama-French 49 industry id |

### 3.5 Ticker/name labels (provenance)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `ticker` | string | 2.8 | Period-supported ticker (null where no verified label) |
| `company_name` | string | 2.8 | Period-supported company name |
| `ticker_name_reference_date` | date32 | 2.8 | Date of the supporting label observation |
| `ticker_name_source` | string | 2.8 | Evidence source for the label |
| `ticker_name_status` | string | 0 | `verified_in_observation_month` (498,706) / `verified_in_filing_on_or_before_observation_date` (15,731) / `no_verified_label_for_observation_period` (14,645) |

### 3.6 THE 147 PREDICTOR CHARACTERISTICS (from `factor_char_list.csv`, all present once)
All `double` except: `age` int64, `f_score` int32, `ni_inc8q` int32. Source: `[A]` = PDF Appendix A Table 1; `[R]` = PDF gap, reconstructed from canonical naming convention. Naming keys: `_me`/`_at`/`_bev`/`_mev`/`_sale` = scaled by market equity / book assets / book EV / market EV / sales; `l1`,`atl1`,`bel1` = lagged denominator; `_gr1` = y-o-y growth; `_gr1a` = annual change ÷ lagged value; `_gr3` = 3y growth; `_21d`,`_126d`,`_252d`,`_1260d` = window in days; `_su` = standardized unexpected; `_chg1` = q-o-q change; `an`/`na` = announcement/non-announcement months.

**Size & valuation**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `market_equity` | Market equity | Banz (1981) [A] | 0 |
| `prc` | Price level | Banz (1981) [R] | 0 |
| `be_me` | Book-to-market equity | Rosenberg, Reid & Lanstein (1985) [A] | 9.0 |
| `bev_mev` | Book-to-market enterprise value | Penman, Richardson & Tuna (2007) [A] | 13.5 |
| `at_me` | Assets-to-market | Fama & French (1992) [A] | 3.2 |
| `ni_me` | Earnings-to-price | Basu (1983) [A] | 3.8 |
| `sale_me` | Sales-to-price | Barbee, Mukherji & Raines (1996) [R] | 3.8 |
| `sale_bev` | Sales-to-book enterprise value | [R] | 11.8 |
| `ocf_me` | Operating cash flow-to-price | Lakonishok, Shleifer & Vishny (1994) [R] | 3.8 |
| `fcf_me` | Free cash flow-to-price | Lakonishok, Shleifer & Vishny (1994) [A] | 4.1 |
| `rd_me` | R&D-to-market equity | Chan, Lakonishok & Sougiannis (2001) [R] | 45.5 |
| `div12m_me` | Dividend yield (12m) | Litzenberger & Ramaswamy (1979) [A] | 6.0 |
| `eqpo_me` | Payout yield | Boudoukh et al. (2007) [A] | 8.7 |
| `eqnpo_me` | Net payout yield | Boudoukh et al. (2007) [A] | 4.0 |
| `eqnpo_12m` | Equity net payout, 12m | Daniel & Titman (2006) [A] | 6.8 |
| `ebitda_mev` | EBITDA-to-market enterprise value | Loughran & Wellman (2011) [A] | 6.7 |
| `ebit_bev` | EBIT-to-book enterprise value | Soliman (2008) [A] | 11.8 |
| `ebit_sale` | EBIT margin (profitability) | Soliman (2008) [A] | 9.0 |

**Investment & growth**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `at_gr1` | Asset growth (1y) | Cooper, Gulen & Schill (2008) [A] | 4.0 |
| `capx_gr1` | CAPEX growth (1y) | Xie (2001) [A] | 9.1 |
| `capx_gr2` | CAPEX growth (2y) | Anderson & Garcia-Feijoo (2006) [A] | 12.5 |
| `capx_gr3` | CAPEX growth (3y) | Anderson & Garcia-Feijoo (2006) [A] | 17.2 |
| `capex_abn` | Abnormal corporate investment | Titman, Wei & Xie (2004) [A] | 21.5 |
| `inv_gr1` | Inventory growth | Belo & Lin (2011) [A] | 36.3 |
| `inv_gr1a` | Inventory change | Thomas & Zhang (2002) [A] | 5.3 |
| `ppeinv_gr1a` | Growth in PP&E + inventory | Davis, Fama & French (2000) [R] | 18.6 |
| `sale_gr1` | Sales growth (1y) | [R] | 10.7 |
| `sale_gr3` | Sales growth (3y) | [R] | 18.2 |
| `debt_gr3` | Growth in book debt (3y) | Lyandres, Sun & Zhang (2008) [A] | 23.6 |
| `sale_emp_gr1` | Sales growth − employment growth | [R] | 15.1 |
| `emp_gr1` | Hiring rate (employment growth) | Belo, Lin & Bazdresch (2014) [A] | 15.3 |

**Balance-sheet change (Richardson et al. 2005 family)**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `be_gr1a` | Change in common equity | Richardson et al. (2005) [A] | 13.1 |
| `coa_gr1a` | Change in current operating assets | Richardson et al. (2005) [A] | 14.3 |
| `col_gr1a` | Change in current operating liabilities | Richardson et al. (2005) [A] | 14.6 |
| `cowc_gr1a` | Change in current operating working capital | Richardson et al. (2005) [A] | 15.0 |
| `ncoa_gr1a` | Change in noncurrent operating assets | Richardson et al. (2005) [A] | 14.3 |
| `ncol_gr1a` | Change in noncurrent operating liabilities | Richardson et al. (2005) [A] | 15.0 |
| `nncoa_gr1a` | Change in net noncurrent operating assets | Richardson et al. (2005) [A] | 15.4 |
| `noa_gr1a` | Change in net operating assets | Hirshleifer et al. (2004) [A] | 15.4 |
| `fnl_gr1a` | Change in financial liabilities | Richardson et al. (2005) [A] | 4.0 |
| `nfna_gr1a` | Change in net financial assets | Richardson et al. (2005) [A] | 4.0 |
| `lti_gr1a` | Change in long-term investments | Richardson et al. (2005) [A] | 9.5 |
| `sti_gr1a` | Change in short-term investments | Richardson et al. (2005) [R] | 5.5 |
| `lnoa_gr1a` | Change in long-term net operating assets | Fairfield, Whisenant & Yohn (2003) [A] | 7.0 |
| `tax_gr1a` | Tax expense surprise | Thomas & Zhang (2011) [A] | 5.0 |

**Profitability & fundamentals**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `gp_at` | Gross profits-to-assets | Novy-Marx (2013) [A] | 3.1 |
| `gp_atl1` | Gross profits-to-lagged assets | [A] | 4.1 |
| `cop_at` | Cash-based operating profits-to-book assets | Ball et al. (2016) [A] | 3.5 |
| `cop_atl1` | Cash-based operating profits-to-lagged assets | Ball et al. (2016) [A] | 4.1 |
| `op_at` | Operating profits-to-assets | Haugen & Baker (1996) [R] | 3.1 |
| `op_atl1` | Operating profits-to-lagged assets | Haugen & Baker (1996) [R] | 4.1 |
| `ope_be` | Operating profits-to-book equity | Fama & French (2015) [R] | 24.8 |
| `ope_bel1` | Operating profits-to-lagged book equity | Fama & French (2015) [R] | 27.0 |
| `opex_at` | Operating expenses-to-assets | [R] | 3.1 |
| `at_turnover` | Capital turnover | Haugen & Baker (1996) [A] | 3.3 |
| `ni_be` | Return on equity (annual) | Haugen & Baker (1996) [A] | 8.9 |
| `niq_at` | Quarterly return on assets | Balakrishnan, Bartov & Faurel (2010) [A] | 3.5 |
| `niq_at_chg1` | Change in quarterly ROA | [A] | 7.7 |
| `niq_be` | Quarterly return on equity | Hou, Xue & Zhang (2015) [A] | 10.4 |
| `niq_be_chg1` | Change in quarterly ROE | [A] | 16.3 |
| `niq_su` | Standardized earnings surprise | Foster, Olsen & Shevlin (1984) [A] | 11.8 |
| `noa_at` | Net operating assets-to-assets | Hirshleifer et al. (2004) [A] | 15.2 |
| `ocf_at` | Operating cash flow-to-assets | Bouchard, Krüger, Landier & Thesmar (2019) [A] | 3.1 |
| `ocf_at_chg1` | Change in OCF-to-assets | Bouchard, Krüger, Landier & Thesmar (2019) [A] | 5.0 |
| `ocfq_saleq_std` | Operating cash flow volatility (std of quarterly OCF/sales) | Bouchard et al. (2019) family [R] | 18.0 |
| `cash_at` | Cash-to-assets | Palazzo (2012) [A] | 2.5 |
| `aliq_at` | Liquidity of book assets | Ortiz-Molina & Phillips (2014) [A] | 14.3 |
| `aliq_mat` | Liquidity of market assets | Ortiz-Molina & Phillips (2014) [A] | 24.2 |
| `tangibility` | Asset tangibility | Hahn & Lee (2009) [A] | 17.1 |
| `age` (int) | Firm age | Jiang, Lee & Zhang (2005) [A] | 0 |
| `intrinsic_value` | Intrinsic value-to-market | Frankel & Lee (1998) [A] | 37.1 |
| `f_score` (int) | Piotroski F-score composite | Piotroski (2000) [A] | 24.8 |
| `o_score` | Ohlson O-score | Dichev (1998) [A] | 16.2 |
| `z_score` | Altman Z-score | Dichev (1998) [A] | 15.5 |
| `kz_index` | Kaplan-Zingales constraint index | Lamont, Polk & Saa-Requejo (2001) [A] | 14.4 |
| `earnings_variability` | Earnings variability | Francis et al. (2004) [A] | 23.2 |
| `ni_ar1` | Earnings persistence (AR1) | Francis et al. (2004) [A] | 23.9 |
| `ni_ivol` | Earnings volatility | Francis et al. (2004) [A] | 23.9 |
| `ni_inc8q` (int) | Consecutive quarters with earnings increases | Barth, Elliott & Finn (1999) [A] | 16.0 |
| `rd_sale` | R&D-to-sales | Chan, Lakonishok & Sougiannis (2001) [R] | 50.2 |
| `rd5_at` | Capitalized 5-year R&D-to-assets | Chan, Lakonishok & Sougiannis (2001) [R] | 56.8 |
| `pi_nix` | Pre-income tax minus income tax | [R] | 42.2 |

**Accruals & issuance**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `oaccruals_at` | Operating accruals-to-assets | Sloan (1996) [A] | 3.5 |
| `oaccruals_ni` | Percent operating accruals | Hafzalla, Lundholm & Van Winkle (2011) [A] | 3.5 |
| `taccruals_at` | Total accruals-to-assets | Sloan (1996) [R] | 4.0 |
| `taccruals_ni` | Total accruals-to-net income | Hafzalla, Lundholm & Van Winkle (2011) [R] | 4.1 |
| `netis_at` | Net total issuance | Bradshaw, Richardson & Sloan (2006) [A] | 3.3 |
| `eqnetis_at` | Net equity issuance | Bradshaw, Richardson & Sloan (2006) [A] | 3.3 |
| `dbnetis_at` | Net debt issuance | Bradshaw, Richardson & Sloan (2006) [A] | 3.1 |
| `chcsho_12m` | Net stock issues (12m common shares) | Pontiff & Woodgate (2008) [A] | 7.2 |

**Leverage & duration**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `at_be` | Book leverage | Fama & French (1992) [A] | 8.4 |
| `debt_me` | Debt-to-market | Bhandari (1988) [A] | 3.2 |
| `netdebt_me` | Net debt-to-price | Penman, Richardson & Tuna (2007) [A] | 3.2 |
| `eq_dur` | Equity duration | Dechow, Sloan & Soliman (2004) [A] | 21.7 |

**Past returns, momentum & seasonality**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `ret_1_0` | Last 1-month return (short-term reversal) | Jegadeesh (1990) [R] | 0.5 |
| `ret_3_1` | Return months −3 to −1 | Jegadeesh & Titman (1993) [R] | 1.6 |
| `ret_6_1` | Return months −6 to −1 | Jegadeesh & Titman (1993) [R] | 3.2 |
| `ret_9_1` | Return months −9 to −1 | Jegadeesh & Titman (1993) [R] | 4.9 |
| `ret_12_1` | Momentum (12−1) | Jegadeesh & Titman (1993) [A] | 6.6 |
| `ret_12_7` | Intermediate momentum (12−7) | Novy-Marx (2012) [R] | 6.7 |
| `ret_60_12` | Long-term reversal (60−12) | De Bondt & Thaler (1985) [R] | 28.9 |
| `resff3_12_1` | Residual momentum 12−1 (FF3 residuals) | Blitz, Huij & Martens (2011) [R] | 13.2 |
| `resff3_6_1` | Residual momentum 6−1 (FF3 residuals) | Blitz, Huij & Martens (2011) [R] | 13.2 |
| `prc_highprc_252d` | Price ÷ 52-week high | George & Hwang (2004) [R] | 3.2 |
| `seas_1_1an` | Seasonality: same calendar month 1y ago (announcement) | Heston & Sadka (2008) [R] | 6.2 |
| `seas_1_1na` | Seasonality: same calendar month 1y ago (non-announcement) | Heston & Sadka (2008) [R] | 6.3 |
| `seas_2_5an` | Seasonality: avg months 2–5y ago (announcement) | Heston & Sadka (2008) [R] | 29.5 |
| `seas_2_5na` | Seasonality: avg months 2–5y ago (non-announcement) | Heston & Sadka (2008) [R] | 29.7 |

**Risk (beta, volatility, skewness)**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `beta_60m` | Market beta (60m) | Fama & MacBeth (1973) [A] | 19.1 |
| `beta_dimson_21d` | Dimson beta (21d) | Dimson (1979) [A] | 0.2 |
| `betabab_1260d` | Frazzini-Pedersen BAB beta (1260d) | Frazzini & Pedersen (2014) [A] | 19.4 |
| `betadown_252d` | Downside beta (252d) | Ang, Chen & Xing (2006) [A] | 3.5 |
| `ivol_capm_21d` | Idio vol, CAPM (21d) | [A] | 0.2 |
| `ivol_capm_252d` | Idio vol, CAPM (252d) | Ali, Hwang & Trombley (2003) [A] | 3.1 |
| `ivol_ff3_21d` | Idio vol, FF3 (21d) | Ang et al. (2006) [A] | 0.9 |
| `ivol_hxz4_21d` | Idio vol, q-factor model (21d) | [A] | 0.9 |
| `iskew_capm_21d` | Idio skewness, CAPM (21d) | [A] | 0.2 |
| `iskew_ff3_21d` | Idio skewness, FF3 (21d) | Bali, Engle & Murray (2016) [A] | 0.9 |
| `iskew_hxz4_21d` | Idio skewness, q-factor (21d) | [A] | 0.9 |
| `coskew_21d` | Coskewness (21d) | Harvey & Siddique (2000) [A] | 0.2 |
| `rvol_21d` | Daily-return volatility (21d) | Bali, Engle & Murray (2016) [R] | 0.2 |
| `rskew_21d` | Daily-return skewness (21d) | Bali, Engle & Murray (2016) [R] | 0.2 |
| `rmax1_21d` | Max daily return (21d) | Bali, Cakici & Whitelaw (2011) [R] | 0.2 |
| `rmax5_21d` | Max 5-day return (21d) | Bali, Cakici & Whitelaw (2011) [R] | 0.2 |
| `rmax5_rvol_21d` | Max 5-day return ÷ vol (21d) | Bali family [R] | 3.2 |
| `corr_1260d` | Correlation with market (1260d) | Asness, Frazzini, Gormsen & Pedersen (2020) [A] | 19.3 |

**Liquidity & trading frictions**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `ami_126d` | Amihud illiquidity measure (126d) | Amihud (2002) [A] | 1.5 |
| `bidaskhl_21d` | High-low bid-ask spread (21d) | Corwin & Schultz (2012) [A] | 0.5 |
| `dolvol_126d` | Dollar trading volume (126d) | Brenner, Chordia & Subrahmanyam (1998) [A] | 1.5 |
| `dolvol_var_126d` | Coefficient of variation, dollar volume (126d) | Chordia, Subrahmanyam & Anshuman (2001) [A] | 1.5 |
| `turnover_126d` | Share turnover (126d) | Datar, Naik & Radcliffe (1998) [A] | 1.5 |
| `turnover_var_126d` | Coefficient of variation, turnover (126d) | Chordia, Subrahmanyam & Anshuman (2001) [A] | 1.5 |
| `zero_trades_21d` | Zero-trading days, 1m (turnover tiebreak) | Liu (2006) [A] | 0.2 |
| `zero_trades_126d` | Zero-trading days, 6m (turnover tiebreak) | Liu (2006) [A] | 1.5 |
| `zero_trades_252d` | Zero-trading days, 12m (turnover tiebreak) | Liu (2006) [A] | 3.1 |

**Composite mispricing / quality**
| Acronym | Description | Reference | Null% |
|---|---|---|---:|
| `mispricing_mgmt` | Mispricing factor: Management | Stambaugh & Yuan (2016) [A] | 4.0 |
| `mispricing_perf` | Mispricing factor: Performance | Stambaugh & Yuan (2016) [A] | 1.4 |
| `qmj` | Quality-minus-junk composite | Asness, Frazzini & Pedersen (2019) [R] | 23.3 |
| `qmj_growth` | QMJ growth component | Asness, Frazzini & Pedersen (2019) [R] | 23.3 |
| `qmj_prof` | QMJ profitability component | Asness, Frazzini & Pedersen (2019) [R] | 3.1 |
| `qmj_safety` | QMJ safety component | Asness, Frazzini & Pedersen (2019) [R] | 2.3 |
| `dgp_dsale` | Δgross margin − Δsales | Abarbanell & Bushee (1998) [A] | 20.7 |
| `dsale_dinv` | Δsales − Δinventory | Abarbanell & Bushee (1998) [A] | 37.7 |
| `dsale_drec` | Δsales − Δreceivables | Abarbanell & Bushee (1998) [A] | 15.0 |
| `dsale_dsga` | Δsales − ΔSG&A | Abarbanell & Bushee (1998) [A] | 27.7 |
| `saleq_gr1` | Quarterly sales growth | [R] | 11.4 |
| `saleq_su` | Standardized unexpected quarterly sales | Foster, Olsen & Shevlin (1984) family [R] | 15.4 |

Count check: 18 (size/valuation) + 13 (investment/growth) + 14 (balance-sheet change) + 35 (profitability/fundamentals) + 8 (accruals/issuance) + 4 (leverage) + 14 (returns/seasonality) + 18 (risk) + 9 (liquidity) + 12 (composite/other) = **149** — two rows appear in two groups (`ebit_sale`/`saleq_gr1` style overlaps). Exact flat count: **147 distinct** (verified against CSV). Use the CSV as the single source of truth for feature selection.

---

## 4. `8k_20150101_20260831_identified.parquet` — ALL 94 COLUMNS MAPPED

### 4.1 Core content
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `document_id` | string | 0 | Unique provider document ID (**not** an SEC accession number) |
| `text` | large_string | 0 | Cleaned main filing text (HTML/whitespace cleaned; wrapper/XBRL artifacts removed; substantive text untouched) |
| `form_type` | string | 0 | `8-K` / `8-K/A` |
| `is_amendment` | bool | 0 | Amendment flag (85 rows true) |
| `form_type_basis` | string | 0 | Evidence for form classification |
| `items` | list\<string\> | 0.3 | Detected 8-K item labels (annotations to verify, not guaranteed exhaustive) |
| `text_chars` / `word_count` | int64 | 0 | Length / approximate word count |
| `text_sha256` | string | 0 | SHA-256 of cleaned text (part of uniqueness key) |

### 4.2 Dates & timing (⚠ provenance caveats)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `filing_date` | date32 | 0 | Provider filing date (unverified vs SEC acceptance) — main availability field |
| `filing_date_raw`, `filing_date_utc_raw` | string | 0 | Original date/time strings |
| `provider_timezone` | string | 0 | Timezone metadata |
| `filing_time_precision` | string | 0 | Always `date_or_midnight_placeholder` |
| `filing_timestamp_utc` | timestamp(us, UTC) | 0 | Provider timestamp — **midnight/date placeholder, not real release time** |
| `sec_accepted_at_utc` | timestamp(us, UTC) | **100** | Schema placeholder (entirely null) |
| `content_report_date` | date32 | 0.3 | Date of report / earliest event from header — **not an availability date**; observed range 2011-11-07→2026-08-31 |
| `content_report_date_raw` / `_basis` | string | 0.3 | Source expression + extraction basis |
| `days_filing_after_content_report` | int32 | 0.3 | filing_date − content_report_date (calendar days) |
| `period_end_date` | date32 | 0 | Provider metadata, not validated |
| `provider_added_at_utc` | timestamp | **100** | Schema placeholder |
| `retrieved_at_utc`, `processed_at_utc`, `research_processed_at_utc`, `identifiers_processed_at_utc` | timestamp | 0 | Package-processing times, **not** historical availability |
| `added_timestamp_raw` | string | 0 | Original provider ingestion string |
| `within_requested_dates` | bool | 0 | Within requested 2015-01-01→2026-08-31 window |
| `archive_year` | int32 | 0 | Provider archive year |

### 4.3 Security identifiers & cross-walk
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `permno` / `permco` | int64 | 0 | CRSP security / company ids |
| `gvkey` / `iid` | string | 0 | Compustat company / issue ids |
| `gvkey_iid` | string | 0 | Convenience key `006066-01` |
| `cusip` / `cusip8` | string | 0 | 9-char CUSIP (incl. check digit) / first 8 chars |
| `cik` | string | 0 | SEC registrant id (10-char; issuer-level, not share class) |
| `isin` | string | **100** | Schema placeholder (all null) |
| `accession_number` | string | **100** | Schema placeholder (all null) |
| `figi`, `composite_figi`, `share_class_figi` | string | 0 / 0 / 0.03 | Original provider FIGI identifiers |
| `cusip_source`, `cusip_asof_date`, `cusip_date_precision` | string / date32 / string | 0 | CUSIP assignment evidence (`monthly_snapshot` 349,181 rows; `continuity_inferred_between_snapshots` 23,958 rows = 2026) |
| `cik_source`, `ticker_source`, `company_name_source` | string | 0 | Identifier evidence sources |
| `identifier_mapping_sources` | list\<string\> | 0 | Mapping evidence sources |
| `identifier_quality_flags` | list\<string\> | 86.9 | e.g. `FIGI_unverified_link_uses_WRDS_ticker_and_SEC_issuer` (898), `filing_CIK_differs_from_Compustat_header_CIK` (1,081), `2026_ticker_continuity_inferred` (72) |
| `identifier_source_row` | int64 | 0 | Upstream row reference |

### 4.4 Link to characteristics panel
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `characteristics_link_date` | date32 | 0 | Date of the verified PERMNO↔GVKEY/IID evidence |
| `characteristics_link_basis` | string | 0 | Basis of the link |
| `compustat_issue_observed_in_filing_month` | bool | 0 | True if the GVKEY-IID is observed in the filing month (370,201 true) |
| `source_file`, `source_row_offset` | string / int64 | 0 | Provenance of the row in the source archive |

### 4.5 Labels & descriptive metadata (⚠ not certified historical)
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `company_name` / `ticker` | string | 0 | Research-period display labels |
| `provider_ticker`, `provider_company_name` | string | 0 | Original provider labels (can disagree with historical labels) |
| `content_company_name` | string | 3.7 | Registrant name parsed from the filing |
| `historical_crsp_company_name` | string | 0 | CRSP name evidence |
| `compustat_header_cusip` / `_cik` / `_ticker` / `_company_name` | string | 0 / 1.3 / 0 / 0 | Current-header Compustat attributes (constant within security; not historical) |
| `openfigi_mapping_cusips` | list\<string\> | 0.2 | FIGI mapping CUSIPs |
| `duplicate_source_document_ids` | list\<string\> | 0 | Equivalent provider IDs retained for provenance |
| `exchange` | string | 0 | Provider exchange code (`UN` NYSE, `UW` Nasdaq) |
| `sector`, `industry`, `country`, `region`, `product_name` | string | 0 | Provider descriptive metadata (not point-in-time certified) |
| `title`, `title_ai` | string | 10.1 / 0 | Provider title / provider AI-generated title (not original filing text) |
| `default_language`, `languages` | string / list\<string\> | 0 | Language metadata |
| `metadata_json` | string | 0 | Raw provider metadata payload |

### 4.6 Cleaning provenance
| Column | Type | Null% | Meaning |
|---|---|---:|---|
| `text_source_field`, `raw_text_sha256`, `source_text_sha256` | string | 0 | Text provenance hashes |
| `cleaner_version`, `source_cleaner_version` | string | 0 | Cleaner versions (research cleaner 2.0.0) |
| `raw_text_chars`, `removed_prefix_chars`, `removed_metadata_chars` | int64 | 0 | Cleaning-size accounting |
| `quality_flags`, `source_quality_flags` | list\<string\> | 0 | Parser/source annotations |
| `cleaning_flags` | list\<string\> | 51.6 | Cleaning annotations (null = none) |
| `source_provider`, `original_source`, `report_url` | string | 0 | Provenance / source URL (not a verified SEC URL) |

### 4.7 Embedded schema metadata (semantics, verbatim keys)
`downloader_version='1.1.0'`, `row_semantics='One provider document/text variant; provider IDs are not SEC accession numbers'`, `timestamp_warning='Provider filing times are unverified; midnight may be placeholder'`, `text_semantics='Full main 8-K provider text after documented wrapper/XBRL cleanup'`, `research_cleaner_version='2.0.0'`, `date_semantics='filing_date remains provider date; content_report_date is event date'`, `identifier_warning='CUSIP/CIK enrichment via historical WRDS identifiers'`.

---

## 5. Model-table mapping (competition spec)

| Role | Column(s) | Note |
|---|---|---|
| Key | `permno` + `eom` (month t) | unique; also (gvkey, iid, eom) |
| Target | `ret_exc_lead1m` | t+1 excess return; splits assigned by target month |
| Features | the 147 CSV variables (+ optional filing-derived aggregates) | case-sensitive; all double except `age`, `f_score`, `ni_inc8q` |
| Universe filters (optional, document if used) | `exch_main`, `common`, `primary_sec`, `shrcd`, `size_grp`, `me`, `prc` | define investable universe deliberately |
| Neutrality inputs | `gics`/`ff49` (sector neutral), `beta_60m` (beta neutral), `size_grp`/`me` (size) | for sector/factor-neutral legs |
| Leg construction | `ret_exc_lead1m` for realized leg returns; `me`/`dolvol` for weighting & borrow-risk screen | short book: avg mkt cap, dollar volume |
| Labels for deck | `ticker`, `company_name` (verified-period labels) | "NVDA, NVIDIA Corporation" convention |
| Join to filings | filings `permno` + `last_day(filing_date)` = chars `eom`, cross-check `gvkey`,`iid` | many-to-one; aggregate filings to stock-month BEFORE join |

**Heavy-missing features to handle explicitly in preprocessing** (>25% null): `rd5_at` 56.8, `rd_sale` 50.2, `pi_nix` 42.2, `rd_me` 45.5, `dsale_dinv` 37.7, `intrinsic_value` 37.1, `inv_gr1` 36.3, `dsale_dsga` 27.7, `seas_2_5na` 29.7, `seas_2_5an` 29.5, `ret_60_12` 28.9, `ope_bel1` 27.0, `ope_be` 24.8, `f_score` 24.8, `aliq_mat` 24.2, `earnings_variability` 23.2, `debt_gr3` 23.6, `qmj_growth`/`qmj` 23.3, `ni_ar1`/`ni_ivol` 23.9, `capex_abn` 21.5, `eq_dur` 21.7, `dgp_dsale` 20.7.
Missing ≠ zero: imputation policy is a team decision and must be documented; do not let target availability define the prediction universe.
