
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  CalendarDays,
  ChevronDown,
  Droplet,
  Download,
  HeartPulse,
  MoreHorizontal,
  Plus,
  Sparkles,
  Target,
  TrendingUp,
  Utensils,
  Zap,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./Dashboard.css";

const glucoseData = [
  { time: "06 AM", glucose: 108 },
  { time: "08 AM", glucose: 124 },
  { time: "10 AM", glucose: 142 },
  { time: "12 PM", glucose: 132 },
  { time: "02 PM", glucose: 158 },
  { time: "04 PM", glucose: 146 },
  { time: "06 PM", glucose: 121 },
  { time: "08 PM", glucose: 136 },
  { time: "10 PM", glucose: 114 },
];

const activityData = [
  { label: "Mon", value: 72 },
  { label: "Tue", value: 85 },
  { label: "Wed", value: 64 },
  { label: "Thu", value: 92 },
  { label: "Fri", value: 78 },
  { label: "Sat", value: 88 },
  { label: "Sun", value: 70 },
];

function MetricCard({ icon: Icon, label, value, unit, change, positive }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <div className="metric-icon">
          <Icon size={19} />
        </div>
        <button className="metric-more" aria-label={`More about ${label}`}>
          <MoreHorizontal size={19} />
        </button>
      </div>

      <p className="metric-label">{label}</p>

      <div className="metric-value">
        {value}
        <span>{unit}</span>
      </div>

      <div className={`metric-change ${positive ? "positive" : "neutral"}`}>
        {positive ? (
          <ArrowDownRight size={15} />
        ) : (
          <ArrowUpRight size={15} />
        )}
        <span>{change}</span>
      </div>
    </div>
  );
}

function Dashboard() {
  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <div className="dashboard-eyebrow">
            <span className="status-dot" />
            HEALTH OVERVIEW
          </div>
          <h1>Your health, <span>in focus.</span></h1>
          <p className="dashboard-subtitle">
            Understand your health patterns and explore your insights.
          </p>
        </div>

        <div className="dashboard-actions">
          <button className="date-filter">
            <CalendarDays size={17} />
            Last 7 days
            <ChevronDown size={15} />
          </button>

          <button className="export-button">
            <Download size={17} />
            Export report
          </button>
        </div>
      </header>

      <div className="demo-notice">
        <Sparkles size={17} />
        <span>
          <strong>Demo dashboard</strong> — All readings and metrics shown here
          are illustrative, not real patient measurements.
        </span>
      </div>

      <section className="metrics-grid">
        <MetricCard
          icon={Droplet}
          label="Average glucose"
          value="128"
          unit="mg/dL"
          change="8% vs previous period"
          positive
        />

        <MetricCard
          icon={Target}
          label="Time in range"
          value="78"
          unit="%"
          change="5% vs previous period"
          positive
        />

        <MetricCard
          icon={HeartPulse}
          label="Average heart rate"
          value="72"
          unit="bpm"
          change="Within demo range"
          positive
        />

        <MetricCard
          icon={Activity}
          label="Daily activity"
          value="6,240"
          unit="steps"
          change="12% vs previous period"
          positive
        />
      </section>

      <section className="dashboard-main-grid">
        <div className="dashboard-panel glucose-panel">
          <div className="panel-heading">
            <div>
              <div className="panel-title-row">
                <div className="panel-icon glucose-icon">
                  <Droplet size={18} />
                </div>
                <h2>Glucose overview</h2>
              </div>
              <p>Illustrative glucose trend throughout the day</p>
            </div>

            <button className="panel-menu" aria-label="Glucose chart options">
              <MoreHorizontal size={20} />
            </button>
          </div>

          <div className="glucose-summary">
            <div>
              <strong>128</strong>
              <span>mg/dL average</span>
            </div>
            <div className="chart-legend">
              <span className="legend-dot" />
              Glucose
            </div>
          </div>

          <div className="glucose-chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={glucoseData}
                margin={{ top: 12, right: 8, left: -18, bottom: 0 }}
              >
                <defs>
                  <linearGradient
                    id="glucoseGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopColor="#159D91" stopOpacity={0.22} />
                    <stop offset="100%" stopColor="#159D91" stopOpacity={0.01} />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  stroke="#edf1f4"
                  strokeDasharray="4 5"
                  vertical={false}
                />

                <XAxis
                  dataKey="time"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#8a96a3", fontSize: 11 }}
                  dy={10}
                />

                <YAxis
                  domain={[60, 200]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#8a96a3", fontSize: 11 }}
                />

                <Tooltip
                  contentStyle={{
                    border: "1px solid #e8edf0",
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                  formatter={(value) => [`${value} mg/dL`, "Glucose"]}
                />

                <Area
                  type="monotone"
                  dataKey="glucose"
                  stroke="#159D91"
                  strokeWidth={2.5}
                  fill="url(#glucoseGradient)"
                  activeDot={{ r: 5, fill: "#159D91" }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-footer">
            <span><span className="range-indicator" /> Illustrative data</span>
            <span>Values are not connected to a glucose monitor</span>
          </div>
        </div>

        <div className="dashboard-panel insights-panel">
          <div className="panel-heading">
            <div>
              <div className="panel-title-row">
                <div className="panel-icon insight-icon">
                  <Sparkles size={18} />
                </div>
                <h2>AI health insights</h2>
              </div>
              <p>Example insights for the demo dataset</p>
            </div>
          </div>

          <div className="insight-card">
            <div className="insight-card-icon">
              <TrendingUp size={18} />
            </div>
            <div>
              <h3>Glucose trend</h3>
              <p>
                The illustrative readings show a rise around the afternoon.
                This is a pattern in demo data, not a personal health finding.
              </p>
            </div>
          </div>

          <div className="insight-card">
            <div className="insight-card-icon food-insight-icon">
              <Utensils size={18} />
            </div>
            <div>
              <h3>Meal tracking</h3>
              <p>
                Connect meal records to explore how logged meals relate to
                glucose measurements.
              </p>
            </div>
          </div>

          <div className="insight-card">
            <div className="insight-card-icon activity-insight-icon">
              <Zap size={18} />
            </div>
            <div>
              <h3>Activity patterns</h3>
              <p>
                Add activity records to compare movement patterns with
                other health measurements.
              </p>
            </div>
          </div>

          <button className="assistant-link">
            Explore with Medical AI Assistant
            <ArrowUpRight size={16} />
          </button>
        </div>
      </section>

      <section className="dashboard-panel activity-panel">
        <div className="panel-heading">
          <div>
            <div className="panel-title-row">
              <div className="panel-icon activity-icon">
                <Activity size={18} />
              </div>
              <h2>Weekly activity</h2>
            </div>
            <p>Illustrative daily activity score</p>
          </div>

          <button className="panel-menu" aria-label="Activity options">
            <MoreHorizontal size={20} />
          </button>
        </div>

        <div className="activity-bars">
          {activityData.map((day) => (
            <div className="activity-day" key={day.label}>
              <div className="activity-bar-track">
                <div
                  className="activity-bar-fill"
                  style={{ height: `${day.value}%` }}
                />
              </div>
              <span>{day.label}</span>
            </div>
          ))}
        </div>
      </section>

      <footer className="dashboard-disclaimer">
        <HeartPulse size={15} />
        <span>
          This dashboard is a software demonstration. It does not provide
          medical diagnoses, treatment recommendations, or real-time monitoring.
        </span>
      </footer>
    </div>
  );
}

export default Dashboard;
