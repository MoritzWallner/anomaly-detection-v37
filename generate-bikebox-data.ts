import { ObjectId } from 'bson';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PROPERTY_ID = 'I6mI0jcMEIMlYXzW2sfx';
const OUTPUT_DIR = path.join(__dirname, 'output', 'mongo');
const METADATA_PATH = path.resolve('/Users/moritzwallner/Downloads/cleaned_dataset/metadata.csv');

// ── Battery → BikeBox mapping ────────────────────────────────────────────────

const BIKEBOX_MAP = [
  { batteryId: 'B0005', bikeboxNum: 1, isAnomaly: false },
  { batteryId: 'B0006', bikeboxNum: 2, isAnomaly: false },
  { batteryId: 'B0007', bikeboxNum: 3, isAnomaly: false },
  { batteryId: 'B0033', bikeboxNum: 4, isAnomaly: false },
  { batteryId: 'B0029', bikeboxNum: 5, isAnomaly: true },
  { batteryId: 'B0034', bikeboxNum: 6, isAnomaly: false },
  { batteryId: 'B0036', bikeboxNum: 7, isAnomaly: false },
] as const;

// Target time range: May 1, 2024 → December 31, 2024
const TARGET_START = new Date('2024-05-01T00:00:00Z').getTime();
const TARGET_END = new Date('2024-12-31T23:59:59Z').getTime();
const TARGET_RANGE = TARGET_END - TARGET_START;

// ── ID generation ──────────────────────────────────────────────────────────────

function genId(prefix: string): string {
  return `${prefix}_${new ObjectId().toHexString()}`;
}

// ── MATLAB date vector parser ────────────────────────────────────────────────
// Handles all format variants:
//   [2010.       7.      21.      15.       0.      35.093]  (trailing dots + spaces)
//   [2.0100e+03 7.0000e+00 ...]                              (scientific notation)
//   [2010    7   21   20   31    5]                           (plain integers)

function parseMatlabDateVector(raw: string): Date {
  const inner = raw.replace(/^\[/, '').replace(/\]$/, '').trim();
  const parts = inner.split(/\s+/).map(s => parseFloat(s));
  const [year, month, day, hour, minute, second] = parts;
  const sec = Math.floor(second);
  const ms = Math.round((second - sec) * 1000);
  return new Date(Date.UTC(year, month - 1, day, hour, minute, sec, ms));
}

// ── CSV parser (simple split — brackets contain no commas) ───────────────────

function parseCSV(filePath: string): Record<string, string>[] {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const record: Record<string, string> = {};
    headers.forEach((h, i) => {
      record[h] = (values[i] ?? '').trim();
    });
    return record;
  });
}

// ── File writer ────────────────────────────────────────────────────────────────

function writeJsonFile(filename: string, data: unknown[]): void {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const filePath = path.join(OUTPUT_DIR, filename);
  const json = data.length > 1000
    ? JSON.stringify(data)
    : JSON.stringify(data, null, 2);
  fs.writeFileSync(filePath, json);
  console.log(`  ${filename}: ${data.length} entries`);
}

// ── Extract discharge data per battery ───────────────────────────────────────

interface DischargePoint {
  timestamp: Date;
  capacity: number;
}

function extractDischargeData(rows: Record<string, string>[], batteryId: string): DischargePoint[] {
  return rows
    .filter(row => row.type === 'discharge' && row.battery_id === batteryId && row.Capacity !== '')
    .map(row => ({
      timestamp: parseMatlabDateVector(row.start_time),
      capacity: parseFloat(row.Capacity),
    }))
    .filter(p => !isNaN(p.capacity) && !isNaN(p.timestamp.getTime()))
    .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
}

// ── Stretch timestamps to May–Dec 2024 ──────────────────────────────────────

function stretchTimestamps(points: DischargePoint[]): DischargePoint[] {
  if (points.length === 0) return [];
  const origMin = points[0].timestamp.getTime();
  const origMax = points[points.length - 1].timestamp.getTime();
  const origRange = origMax - origMin || 1;

  return points.map(p => ({
    timestamp: new Date(TARGET_START + ((p.timestamp.getTime() - origMin) / origRange) * TARGET_RANGE),
    capacity: p.capacity,
  }));
}

// ── Main ───────────────────────────────────────────────────────────────────────

function main(): void {
  console.log('Generating BikeBox MongoDB data...\n');

  const rows = parseCSV(METADATA_PATH);
  console.log(`  Loaded ${rows.length} rows from metadata.csv\n`);

  // Extract and stretch data for each bikebox
  const bikeboxData = new Map<number, {
    parameterId: string;
    points: DischargePoint[];
    isAnomaly: boolean;
    batteryId: string;
  }>();

  for (const bb of BIKEBOX_MAP) {
    const raw = extractDischargeData(rows, bb.batteryId);
    const stretched = stretchTimestamps(raw);
    const parameterId = genId('pmt');
    bikeboxData.set(bb.bikeboxNum, {
      parameterId,
      points: stretched,
      isAnomaly: bb.isAnomaly,
      batteryId: bb.batteryId,
    });
    console.log(`  BikeBox ${bb.bikeboxNum} (${bb.batteryId}): ${stretched.length} discharge cycles`);
  }
  console.log('');

  // ── 1. Parameters (7) ────────────────────────────────────────────────────

  const parameters = BIKEBOX_MAP.map(bb => {
    const data = bikeboxData.get(bb.bikeboxNum)!;
    const capacities = data.points.map(p => p.capacity);
    const lastCapacity = capacities[capacities.length - 1] ?? 0;
    const minCap = Math.min(...capacities);
    const maxCap = Math.max(...capacities);

    return {
      _id: data.parameterId,
      object: 'parameter',
      objectType: 'dto',
      version: 1,
      propertyId: PROPERTY_ID,
      name: `BikeBox ${bb.bikeboxNum} Capacity`,
      description: null,
      history: true,
      type: 'number',
      currentValue: lastCapacity,
      unit: 'ampere',
      min: parseFloat((minCap - 0.1).toFixed(4)),
      max: parseFloat((maxCap + 0.1).toFixed(4)),
      lastUpdatedAt: { $date: data.points[data.points.length - 1]?.timestamp.toISOString() ?? new Date().toISOString() },
      createdAt: { $date: data.points[0]?.timestamp.toISOString() ?? new Date().toISOString() },
      reference: null,
    };
  });

  writeJsonFile('bikebox_parameters.json', parameters);

  // ── 2. ParameterHistories (~1135) ────────────────────────────────────────

  const histories: unknown[] = [];
  for (const bb of BIKEBOX_MAP) {
    const data = bikeboxData.get(bb.bikeboxNum)!;
    for (const point of data.points) {
      histories.push({
        _id: genId('pmth'),
        version: 1,
        createdAt: { $date: point.timestamp.toISOString() },
        currentValue: point.capacity,
        static: {
          type: 'number',
          propertyId: PROPERTY_ID,
          reference: null,
          parameterId: data.parameterId,
        },
      });
    }
  }

  writeJsonFile('bikebox_parameterHistories.json', histories);

  // ── 3. ParameterAnomalyGroupTracker (1) ──────────────────────────────────

  const trackerId = genId('pmtagt');
  const tracker = [{
    _id: trackerId,
    object: 'parameterAnomalyGroupTracker',
    objectType: 'dto',
    version: 1,
    propertyId: PROPERTY_ID,
    name: 'BikeBox Battery Monitoring',
    featureReferenceArray: ['capacity'],
    description: null,
    dataType: 'time-series',
    nextRunAt: null,
  }];

  writeJsonFile('bikebox_parameterAnomalyGroupTracker.json', tracker);

  // ── 4. ParameterAnomalyGroups (7) ────────────────────────────────────────

  const groups = BIKEBOX_MAP.map(bb => {
    const data = bikeboxData.get(bb.bikeboxNum)!;
    const capacities = data.points.map(p => p.capacity);

    // Temporal features: [slope, maxDrop, avgStd]
    const slope = capacities.length > 1
      ? (capacities[capacities.length - 1] - capacities[0]) / (capacities.length - 1)
      : 0;

    let maxDrop = 0;
    for (let i = 1; i < capacities.length; i++) {
      const drop = capacities[i - 1] - capacities[i];
      if (drop > maxDrop) maxDrop = drop;
    }

    const windowSize = 10;
    const stds: number[] = [];
    for (let i = 0; i <= capacities.length - windowSize; i++) {
      const window = capacities.slice(i, i + windowSize);
      const mean = window.reduce((a, b) => a + b, 0) / window.length;
      const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / window.length;
      stds.push(Math.sqrt(variance));
    }
    const avgStd = stds.length > 0 ? stds.reduce((a, b) => a + b, 0) / stds.length : 0;

    const temporalFeatures: [number, number, number] = [
      parseFloat(slope.toFixed(6)),
      parseFloat(maxDrop.toFixed(6)),
      parseFloat(avgStd.toFixed(6)),
    ];

    const isOutlier = bb.isAnomaly;
    const anomalyScore = isOutlier ? 0.95 : parseFloat((0.03 + bb.bikeboxNum * 0.02).toFixed(4));

    return {
      _id: genId('pmtag'),
      object: 'parameterAnomalyGroup',
      objectType: 'dto',
      version: 1,
      parameterAnomalyGroupTrackerId: trackerId,
      propertyId: PROPERTY_ID,
      name: `BikeBox ${bb.bikeboxNum}`,
      anomalyScore,
      isOutlier,
      incidentId: null,
      featureReferenceParameterIdArray: [{
        parameterId: data.parameterId,
        featureReference: 'capacity',
        temporalFeatures,
        isOutlier,
        unit: 'ampere',
      }],
      anomalyDescription: isOutlier
        ? 'BikeBox 5 shows significantly accelerated battery capacity degradation compared to the fleet. The battery (tested at 43°C elevated temperature) exhibits a much steeper capacity decline over fewer discharge cycles, indicating thermal stress-induced degradation. This anomalous wear pattern suggests the battery is operating outside normal thermal conditions and may require early replacement.'
        : null,
    };
  });

  writeJsonFile('bikebox_parameterAnomalyGroups.json', groups);

  console.log('\nDone! Files written to output/mongo/');
}

main();
