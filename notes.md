Please find the device details below for your reference.
IMEI                           Model
863738070450771 FMC920
863738070456042 FMC920
863738070458055 FMC920
863738070513768 FMC920
863738070513826 FMC920
863738070534061 FMC920
863738070627071 FMC920
863738070627121 FMC920
863738070627303 FMC920
863738070627741 FMC920
863738070666889 FMC920
863738070727442 FMC920
863738070729430 FMC920
863738070731550 FMC920
863738070734950 FMC920
863738070735353 FMC920
863738070738233 FMC920
863738070803409 FMC920
863738070810727 FMC920
863738070810867 FMC920
863738070810966 FMC920
863738070811410 FMC920
863738070811857 FMC920
863738070812137 FMC920
863738070812640 FMC920

## Questions 

### Questions for Domain Experts - GPS Data Quality & Cleaning Decisions
### Section 1: Geographic Boundaries & Out-of-Bounds Data
Context: Some GPS points fall outside IITM campus boundaries.

Do buses ever legitimately leave campus?
Are there routes that go outside IITM (e.g., to nearby areas, hostels outside campus)?
If yes, what are the typical external destinations?
Should we expand our boundary box or keep strict campus-only bounds?
How should we handle out-of-bounds points?
Option A: Remove all out-of-bounds points completely
Option B: Keep them but flag as "external route"
Option C: Remove only if they're far away (e.g., >1 km from campus)
Which approach makes sense for your use case?
Are there specific entry/exit gates we should know about?
This helps distinguish legitimate exits vs GPS errors

### Section 2: Temporal Data Issues
Context: Found transmission delays, out-of-order timestamps, and sampling gaps.

What is acceptable transmission delay?
Is 30-second delay acceptable for real-time ETA predictions?
Should we discard data with delays > X seconds?
Or should we use the GPS timestamp (DateTime) and ignore packet received time?
How should we handle data gaps?
If a bus has no data for 5+ minutes, should we:
Option A: Treat as separate trips
Option B: Interpolate the missing data
Option C: Mark as "data loss period" and exclude from training
What gap duration indicates a new trip vs temporary signal loss?
Are there known maintenance/downtime periods?
Specific times when GPS devices are turned off?
This helps distinguish intentional gaps from data quality issues

### Section 3: Speed Anomalies
Context: Found speeds >60 km/h, speed=0 with position changes, and speed>0 with static position.

What are realistic speed limits on campus?
Maximum speed buses should travel?
Typical speeds in different areas (main roads vs narrow lanes)?
Should we cap speeds at a certain threshold (e.g., 50 km/h)?
When speed sensor and GPS disagree, which is more reliable?
Example: Speed=0 but coordinates changing
Should we trust:
Option A: Speed sensor (maybe GPS drift)
Option B: GPS position (maybe speed sensor lag)
Option C: Calculate speed from position and override sensor
Are there areas where buses frequently stop?
Known bus stops, traffic signals, speed bumps?
This helps validate stationary periods

### Section 4: Stationary Periods & Stops
Context: GPS coordinates "drift" even when buses are stationary.

How much GPS drift is acceptable when stationary?
Should we consider position changes <5 meters as "stationary drift"?
Or use a stricter threshold?
What defines a "bus stop" vs just waiting in traffic?
Minimum dwell time to count as a stop (e.g., >30 seconds)?
Do you have a list of official bus stop locations?
Should we auto-detect stops or use predefined locations?
Should we filter out overnight/parking data?
If buses are parked 11 PM - 6 AM, should we exclude this data?
Or keep it to understand depot locations?
🚍 Section 5: Bus/IMEI-Specific Issues
Context: Some IMEIs have very little data, some have thousands of records.

Are all IMEIs active buses?
Could some be test devices, decommissioned buses, or non-bus vehicles?
Should we exclude IMEIs with <X records per day?
Do different buses serve different routes?
Are routes fixed per bus or do buses rotate?
This affects how we model route-specific patterns
Are there known faulty GPS devices?
Specific IMEIs with known hardware issues?
Should we exclude them entirely?

### Section 6: Data Cleaning Strategy
Context: Need to decide on overall cleaning approach.

What is the primary use case for this data?
Real-time ETA prediction for students?
Historical route analysis?
Fleet management?
(This determines how aggressive we should be with cleaning)
How much data loss is acceptable?
If cleaning removes 10% of data, is that okay?
Or should we be conservative and keep questionable data?
Should we create multiple datasets?
Dataset A: Strictly cleaned (high quality, less data)
Dataset B: Loosely cleaned (more data, some noise)
Dataset C: Raw (no cleaning)
Then compare model performance?

### Section 7: Feature Engineering Priorities
Context: Planning to extract features from minimal columns.

What are the most important factors for ETA prediction?
Time of day? Day of week? Weather? Traffic? Route congestion?
This helps prioritize which features to engineer
Are there external factors we should consider?
Class schedules (more traffic during class changes)?
Events (convocations, fests)?
Weather data available?
Exam periods?
Do you have historical ETA data or ground truth?
Actual arrival times at stops?
This helps validate our predictions

### Section 8: Real-Time System Requirements
Context: Building a production real-time ETA system.

What is the acceptable prediction latency?
Predictions must be generated within X seconds?
This affects feature complexity
How often should predictions update?
Every second? Every 10 seconds? Every minute?
Affects data processing pipeline
What happens when a bus has no recent data?
Show "GPS unavailable"?
Use last known position?
Use historical average?


### lat, log
velachery gate: Lat=12.988354, Lon=80.223556
hostal: Lat=12.986556, Lon=80.238536
main gate: Lat=13.006035, Lon=80.241863

```python
const busStops = [
  { name: 'Velachery Gate IIT Campus Stop', lat: 12.988763557585395, lng: 80.22367960515129, id: 1 },
  { name: 'NAC2 Stop Towards GC', lat: 12.990236, lng: 80.227548, id: 2 },
  { name: 'Nac2 Towards Velachery Stop', lat: 12.9898636, lng: 80.2271109, id: 3 },
  { name: 'CRC Towards GC Stop', lat: 12.99089, lng: 80.230274, id: 4 },
  { name: 'CRC Towards Velachery Stop', lat: 12.99079680, lng: 80.23016660, id: 5 },  
  { name: 'HSB Stop Towards Velachery Stop', lat: 12.99076780, lng: 80.23168950, id: 6},
  { name: 'HSB Stop Towards GC', lat: 12.991037, lng: 80.232065, id: 7},
  { name: 'KV Stop Towards Main Gate', lat: 12.991868573807471, lng: 80.23367454352295, id: 8 },
  { name: 'KV Stop Towards GC', lat: 12.991910, lng: 80.233785, id: 9 },
  { name: 'Postoffice Stop Towards Main Gate', lat: 12.99388680, lng: 80.23428760, id: 10 },
  { name: 'Postoffice Stop Towards GC', lat: 12.99393340, lng: 80.23459350, id: 11 },
  { name: 'E1 Stop Towards Main Gate', lat: 12.9960173, lng: 80.2359173, id: 12 },
  { name: 'E1 Stop Towards GC', lat: 12.996117, lng: 80.236184, id: 13},
  { name: 'Vana Vani Stop Towards Main Gate', lat: 12.998704, lng: 80.2391799, id: 14 },
  { name: 'Vana Vani Stop Towards GC', lat: 12.999084,   lng:  80.239380, id: 15 },
  { name: 'D1 Stop Towards Main Gate', lat:13.002546, lng: 80.240091, id: 16},
  { name: 'D1 Stop Towards GC', lat: 13.002679,  lng: 80.240219, id: 17 },
  { name: 'Main Gate', lat: 13.00612964850378, lng: 80.24191299117685, id: 18 },
  { name: 'GC Stop Towards Hostel', lat: 12.991342, lng: 80.233637, id: 19 },
  { name: 'Library Stop', lat: 12.9907517, lng: 80.2334554, id: 20 },
  { name: 'OAT Stop Towards Hostel', lat: 12.989257,  lng: 80.233031, id: 21},
  { name: 'Gymkhana Stop Towards GC', lat: 12.9866372, lng: 80.2332951, id: 22},
  { name: 'Gymkhana Stop Towards Hostel', lat: 12.986615, lng: 80.233366, id: 23 },
  { name: 'Narmada Stop Towards GC', lat: 12.9862759, lng: 80.2350107, id: 24 },
  { name: 'Narmada Stop Towards Hostel', lat: 12.986546, lng: 80.235301, id: 25 },
  { name: 'Jamuna and Ganga Stand', lat: 12.986566332825637, lng: 80.23855704439099, id: 26 },
  { name: 'RP Stand', lat: 12.990088, lng: 80.241799, id: 28},
  { name: 'ED Stand', lat: 12.989874633072303, lng: 80.22646042038593, id: 29},
];

// Route sequences by stop id
const Hostel_to_Main   = [26, 24, 22, 20, 6, 5, 3, 1, 2, 4, 7, 8, 10, 12, 14, 16, 18];
const Main_to_Hostel   = [18, 17, 15, 13, 11, 9, 6, 5, 3, 1, 2, 4, 7, 19, 21, 23, 25, 26];
const Velachery_to_Main= [1, 2, 4, 7, 8, 10, 12, 14, 16, 18];
const Velachery_to_Hostel=[1, 2, 4, 7, 19, 21, 23, 25, 26];
const Hotel_to_Velachery=[26, 24, 22, 20, 6, 5, 3, 1];
const Main_to_Velachery=[18, 17, 15, 13, 11, 9, 6, 5, 3, 1];
const RP_to_ED         = [28, 22, 21, 20, 6, 5, 3, 29];
const ED_to_RP         = [29, 2, 4, 7, 19, 21, 23, 28];

const ROUTES = [
  { code: "HOSTEL_MAIN", label: "Hostel → Main Gate", ids: Hostel_to_Main },
  { code: "MAIN_HOSTEL", label: "Main Gate → Hostel", ids: Main_to_Hostel },
  { code: "VG_MAIN",     label: "Velachery Gate → Main Gate", ids: Velachery_to_Main },
  { code: "VG_HOSTEL",   label: "Velachery Gate → Hostel", ids: Velachery_to_Hostel },
  { code: "MAIN_VG",     label: "Main Gate → Velachery Gate", ids: Main_to_Velachery },
  { code: "HOSTEL_VG",   label: "Hostel → Velachery Gate", ids: Hotel_to_Velachery },
  { code: "RP_ED",       label: "Research Park → ED", ids: RP_to_ED },
  { code: "ED_RP",       label: "ED → Research Park", ids: ED_to_RP },
];
```

```python
const HOSTEL_TO_MAIN = [
  '06:20 AM','06:40 AM','07:00 AM','07:20 AM','07:40 AM','08:00 AM','08:20 AM','08:40 AM',
  '09:00 AM','09:20 AM','09:40 AM','10:00 AM','10:20 AM','10:40 AM','11:00 AM','11:20 AM',
  '11:40 AM','12:00 PM','12:20 PM','12:40 PM','01:00 PM','01:20 PM','01:40 PM','02:00 PM',
  '02:20 PM','02:40 PM','03:00 PM','03:20 PM','03:40 PM','04:00 PM','04:20 PM','04:40 PM',
  '05:00 PM','05:20 PM','05:40 PM','06:00 PM','06:20 PM','06:40 PM','07:00 PM','07:20 PM',
  '07:40 PM','08:00 PM','08:20 PM','08:40 PM','09:00 PM','09:20 PM'
];
const HOSTEL_NOTES = [
  { time: '09:30 PM', note: 'To Velachery Gate' },
  { time: 'bustransport@iitm.ac.in', note: 'For your feedback/complaints, please send a mail here.' }
];
const MAIN_TO_HOSTEL = [...HOSTEL_TO_MAIN];
const MAIN_NOTES = [
  { time: '09:40 PM', note: 'To Velachery Gate' },
  { time: '10:00 PM', note: 'To Velachery Gate' },
  { time: 'bustransport@iitm.ac.in', note: 'For your feedback/complaints, please send a mail here.' }
];
const VELACHERY_TO_MAIN = [
  '06:15 AM','06:35 AM','06:55 AM','07:15 AM','07:35 AM','07:55 AM','08:15 AM','08:35 AM','08:55 AM',
  '09:15 AM','09:35 AM','09:55 AM','10:15 AM','10:35 AM','10:55 AM','11:15 AM','11:35 AM',
  '12:15 PM','12:35 PM','12:55 PM','01:15 PM','01:35 PM','01:55 PM','02:15 PM','02:35 PM','02:55 PM',
  '03:15 PM','03:35 PM','03:55 PM','04:15 PM','04:35 PM','04:55 PM','05:15 PM','05:35 PM','05:55 PM',
  '06:15 PM','06:35 PM','06:55 PM','07:15 PM','07:35 PM','07:55 PM','08:15 PM','08:35 PM','08:55 PM',
  '09:15 PM','09:35 PM'
];
const VELACHERY_TO_HOSTEL = [
  '06:25 AM','06:45 AM','07:05 AM','07:25 AM','07:45 AM','08:05 AM','08:25 AM','08:45 AM','09:05 AM',
  '09:25 AM','09:45 AM','10:05 AM','10:25 AM','10:45 AM','11:05 AM','11:25 AM','11:45 AM',
  '12:05 PM','12:25 PM','12:45 PM','01:05 PM','01:25 PM','01:45 PM','02:05 PM','02:25 PM','02:45 PM',
  '03:05 PM','03:25 PM','03:45 PM','04:05 PM','04:25 PM','04:45 PM','05:05 PM','05:25 PM','05:45 PM',
  '06:05 PM','06:25 PM','06:45 PM','07:05 PM','07:25 PM','07:45 PM','08:05 PM','08:25 PM','08:45 PM',
  '09:05 PM','09:15 PM'
];
const VELACHERY_NOTES = [
  { time: 'bustransport@iitm.ac.in', note: 'For your feedback/complaints, please send a mail here.' }
];
const RP_TO_ED_TIMES = [
  '07:30 AM','08:05 AM','08:40 AM','09:00 AM','09:15 AM','09:35 AM','09:45 AM',
  '10:15 AM','10:25 AM','10:45 AM','11:20 AM','11:40 AM',
  '12:05 PM','12:45 PM','01:40 PM','02:20 PM','02:35 PM','03:00 PM','03:15 PM','03:45 PM',
  '04:15 PM','04:50 PM','05:05 PM','05:50 PM','06:25 PM','07:05 PM'
];
const ED_TO_RP_TIMES = [
  '07:50 AM','08:25 AM','09:00 AM','09:15 AM','09:30 AM','10:00 AM','10:05 AM','10:30 AM',
  '11:00 AM','11:15 AM','11:45 AM',
  '12:00 PM','12:25 PM','01:05 PM','02:00 PM','02:40 PM','03:00 PM','03:20 PM','03:30 PM',
  '04:05 PM','04:45 PM','05:10 PM','05:30 PM','06:10 PM','06:45 PM','07:20 PM'
];
const Cart1 = [['07:30','07:50'],['08:05','08:25'],['08:40','09:00'],['09:15','09:30'],['09:45','10:05'],['10:25','11:00'],['11:20','11:45'],['12:05','12:25'],['13:40','14:00'],['14:20','14:40'],['15:00','15:20'],['15:45','16:05'],['16:50','17:10']];
const Cart3 = [['09:00','09:15'],['09:35','10:00'],['10:15','10:30'],['10:45','11:15'],['11:40','12:00'],['12:45','13:05'],['14:35','15:00'],['15:15','15:30'],['16:15','16:45'],['17:05','17:30'],['17:50','18:10'],['18:25','18:45'],['19:05','19:20']];
```