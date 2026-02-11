import { ObjectId } from 'bson';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PROPERTY_ID = 'I6mI0jcMEIMlYXzW2sfx';
const OUTPUT_DIR = path.join(__dirname, 'output', 'mongo');

// ── ID generation ──────────────────────────────────────────────────────────────

function genId(prefix: string): string {
  return `${prefix}_${new ObjectId().toHexString()}`;
}

// ── Date helpers ───────────────────────────────────────────────────────────────

function toMongoDate(dateStr: string): { $date: string } {
  const iso = dateStr.replace(' ', 'T') + (dateStr.includes('.') ? 'Z' : '.000Z');
  return { $date: new Date(iso).toISOString() };
}

// ── CSV parser (simple split — no quoted fields in these datasets) ─────────────

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
  // Pretty-print small files, compact for large ones
  const json = data.length > 1000
    ? JSON.stringify(data)
    : JSON.stringify(data, null, 2);
  fs.writeFileSync(filePath, json);
  console.log(`  ${filename}: ${data.length} entries`);
}

// ── Traffic (histories only) ───────────────────────────────────────────────────

function generateTrafficHistories(): void {
  console.log('Traffic:');

  const junctionParamId: Record<string, string> = {
    '1': 'pmt_698c4b05083ca3a2a36199ae',
    '2': 'pmt_698c4b0f083ca3a2a36199bc',
    '3': 'pmt_698c4b5d083ca3a2a3619a1a',
    '4': 'pmt_698c4b69083ca3a2a3619a28',
  };

  const rows = parseCSV(path.join(__dirname, 'datasets', 'traffic.csv'));

  const histories = rows.map(row => ({
    _id: genId('pmth'),
    version: 1,
    createdAt: toMongoDate(row.DateTime),
    currentValue: parseFloat(row.Vehicles),
    static: {
      type: 'number',
      propertyId: PROPERTY_ID,
      reference: null,
      parameterId: junctionParamId[row.Junction],
    },
  }));

  writeJsonFile('traffic_parameterHistories.json', histories);
}

// ── Vehicles (histories only) ──────────────────────────────────────────────────

function generateVehiclesHistories(): void {
  console.log('Vehicles:');

  const paramId: Record<string, string> = {
    'CUP1_voltage': 'pmt_698c5c6622bf799d09410df5',
    'CUP1_soc': 'pmt_698c5c6622bf799d09410df6',
    'CUP2_voltage': 'pmt_698c5c6622bf799d09410df7',
    'CUP2_soc': 'pmt_698c5c6622bf799d09410df8',
    'CUP3_voltage': 'pmt_698c5c6622bf799d09410df9',
    'CUP3_soc': 'pmt_698c5c6622bf799d09410dfa',
    'CUP4_voltage': 'pmt_698c5c6622bf799d09410dfb',
    'CUP4_soc': 'pmt_698c5c6622bf799d09410dfc',
    'CUP5_voltage': 'pmt_698c5c6622bf799d09410dfd',
    'CUP5_soc': 'pmt_698c5c6622bf799d09410dfe',
    'ID1_voltage': 'pmt_698c5c6622bf799d09410dff',
    'ID1_soc': 'pmt_698c5c6622bf799d09410e00',
    'ID2_voltage': 'pmt_698c5c6622bf799d09410e01',
    'ID2_soc': 'pmt_698c5c6622bf799d09410e02',
  };

  const rows = parseCSV(path.join(__dirname, 'datasets', 'vehicles.csv'));
  const features = ['voltage', 'soc'] as const;

  const histories = rows.flatMap(row =>
    features.map(feature => ({
      _id: genId('pmth'),
      version: 1,
      createdAt: toMongoDate(row.time),
      currentValue: parseFloat(row[feature]),
      static: {
        type: 'number',
        propertyId: PROPERTY_ID,
        reference: null,
        parameterId: paramId[`${row.vehicle}_${feature}`],
      },
    }))
  );

  writeJsonFile('vehicles_parameterHistories.json', histories);
}

// ── Customers (histories only) ─────────────────────────────────────────────────

function generateCustomersHistories(): void {
  console.log('Customers:');

  const paramId: Record<string, string> = {
    'CUST_001_avg_transaction': 'pmt_698c5c6722bf799d0945e95d',
    'CUST_001_monthly_logins': 'pmt_698c5c6722bf799d0945e95e',
    'CUST_001_support_tickets': 'pmt_698c5c6722bf799d0945e95f',
    'CUST_001_account_age_days': 'pmt_698c5c6722bf799d0945e960',
    'CUST_002_avg_transaction': 'pmt_698c5c6722bf799d0945e961',
    'CUST_002_monthly_logins': 'pmt_698c5c6722bf799d0945e962',
    'CUST_002_support_tickets': 'pmt_698c5c6722bf799d0945e963',
    'CUST_002_account_age_days': 'pmt_698c5c6722bf799d0945e964',
    'CUST_003_avg_transaction': 'pmt_698c5c6722bf799d0945e965',
    'CUST_003_monthly_logins': 'pmt_698c5c6722bf799d0945e966',
    'CUST_003_support_tickets': 'pmt_698c5c6722bf799d0945e967',
    'CUST_003_account_age_days': 'pmt_698c5c6722bf799d0945e968',
    'CUST_004_avg_transaction': 'pmt_698c5c6722bf799d0945e969',
    'CUST_004_monthly_logins': 'pmt_698c5c6722bf799d0945e96a',
    'CUST_004_support_tickets': 'pmt_698c5c6722bf799d0945e96b',
    'CUST_004_account_age_days': 'pmt_698c5c6722bf799d0945e96c',
    'CUST_005_avg_transaction': 'pmt_698c5c6722bf799d0945e96d',
    'CUST_005_monthly_logins': 'pmt_698c5c6722bf799d0945e96e',
    'CUST_005_support_tickets': 'pmt_698c5c6722bf799d0945e96f',
    'CUST_005_account_age_days': 'pmt_698c5c6722bf799d0945e970',
    'CUST_006_avg_transaction': 'pmt_698c5c6722bf799d0945e971',
    'CUST_006_monthly_logins': 'pmt_698c5c6722bf799d0945e972',
    'CUST_006_support_tickets': 'pmt_698c5c6722bf799d0945e973',
    'CUST_006_account_age_days': 'pmt_698c5c6722bf799d0945e974',
    'CUST_007_avg_transaction': 'pmt_698c5c6722bf799d0945e975',
    'CUST_007_monthly_logins': 'pmt_698c5c6722bf799d0945e976',
    'CUST_007_support_tickets': 'pmt_698c5c6722bf799d0945e977',
    'CUST_007_account_age_days': 'pmt_698c5c6722bf799d0945e978',
    'CUST_008_avg_transaction': 'pmt_698c5c6722bf799d0945e979',
    'CUST_008_monthly_logins': 'pmt_698c5c6722bf799d0945e97a',
    'CUST_008_support_tickets': 'pmt_698c5c6722bf799d0945e97b',
    'CUST_008_account_age_days': 'pmt_698c5c6722bf799d0945e97c',
    'CUST_009_avg_transaction': 'pmt_698c5c6722bf799d0945e97d',
    'CUST_009_monthly_logins': 'pmt_698c5c6722bf799d0945e97e',
    'CUST_009_support_tickets': 'pmt_698c5c6722bf799d0945e97f',
    'CUST_009_account_age_days': 'pmt_698c5c6722bf799d0945e980',
    'CUST_010_avg_transaction': 'pmt_698c5c6722bf799d0945e981',
    'CUST_010_monthly_logins': 'pmt_698c5c6722bf799d0945e982',
    'CUST_010_support_tickets': 'pmt_698c5c6722bf799d0945e983',
    'CUST_010_account_age_days': 'pmt_698c5c6722bf799d0945e984',
  };

  const rows = parseCSV(path.join(__dirname, 'datasets', 'customers.csv'));
  const features = ['avg_transaction', 'monthly_logins', 'support_tickets', 'account_age_days'] as const;

  const now = { $date: new Date().toISOString() };

  const histories = rows.flatMap(row =>
    features.map(feature => ({
      _id: genId('pmth'),
      version: 1,
      createdAt: now,
      currentValue: parseFloat(row[feature]),
      static: {
        type: 'number',
        propertyId: PROPERTY_ID,
        reference: null,
        parameterId: paramId[`${row.customer_id}_${feature}`],
      },
    }))
  );

  writeJsonFile('customers_parameterHistories.json', histories);
}

// ── Main ───────────────────────────────────────────────────────────────────────

function main(): void {
  console.log('Generating MongoDB data...\n');

  generateTrafficHistories();
  console.log('');
  generateVehiclesHistories();
  console.log('');
  generateCustomersHistories();

  console.log('\nDone! Files written to output/mongo/');
}

main();
