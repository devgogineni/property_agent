CREATE INDEX IF NOT EXISTS idx_hpi_region_name ON hpi(region_name);
CREATE INDEX IF NOT EXISTS idx_hpi_region_date  ON hpi(region_name, period_date DESC);
CREATE INDEX IF NOT EXISTS idx_hpi_date         ON hpi(period_date DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ppd_unique_id ON ppd(unique_id);
CREATE INDEX IF NOT EXISTS idx_ppd_postcode    ON ppd(postcode);
CREATE INDEX IF NOT EXISTS idx_ppd_town_street ON ppd(town, street);
CREATE INDEX IF NOT EXISTS idx_ppd_locality    ON ppd(locality);
CREATE INDEX IF NOT EXISTS idx_ppd_district    ON ppd(district);
CREATE INDEX IF NOT EXISTS idx_ppd_county       ON ppd(county);
CREATE INDEX IF NOT EXISTS idx_ppd_deed_date   ON ppd(deed_date DESC);
