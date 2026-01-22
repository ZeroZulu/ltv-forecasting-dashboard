import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Area, AreaChart, PieChart, Pie, Cell } from 'recharts';

// LTV Data
const channelData = [
  { name: 'Referral', shortName: 'REF', avgLTV: 714, cac: 5, players: 1526, roi: 143, retention: 0.24 },
  { name: 'Organic Search', shortName: 'ORG', avgLTV: 518, cac: 0.01, players: 2035, roi: 165, retention: 0.25 },
  { name: 'Cross Promo', shortName: 'XPR', avgLTV: 499, cac: 3, players: 961, roi: 166, retention: 0.24 },
  { name: 'Influencer', shortName: 'INF', avgLTV: 455, cac: 25, players: 1479, roi: 18, retention: 0.24 },
  { name: 'Paid Social', shortName: 'SOC', avgLTV: 429, cac: 12.5, players: 2547, roi: 34, retention: 0.24 },
  { name: 'App Store', shortName: 'APP', avgLTV: 341, cac: 8, players: 1452, roi: 43, retention: 0.24 },
];

const monthlyData = [
  { month: 'Jan', ltv: 180, revenue: 420000 },
  { month: 'Feb', ltv: 220, revenue: 480000 },
  { month: 'Mar', ltv: 310, revenue: 520000 },
  { month: 'Apr', ltv: 380, revenue: 610000 },
  { month: 'May', ltv: 520, revenue: 780000 },
  { month: 'Jun', ltv: 478, revenue: 720000 },
  { month: 'Jul', ltv: 450, revenue: 680000 },
  { month: 'Aug', ltv: 490, revenue: 740000 },
  { month: 'Sep', ltv: 510, revenue: 760000 },
];

const playerData = [
  { id: 'PLY-847291', channel: 'Referral', ltv: '$1,847.23', segment: 'Whale', status: 'Active', date: 'Jan 15' },
  { id: 'PLY-293847', channel: 'Paid Social', ltv: '$892.45', segment: 'Dolphin', status: 'Active', date: 'Jan 14' },
  { id: 'PLY-192837', channel: 'Organic', ltv: '$234.12', segment: 'Minnow', status: 'Churned', date: 'Jan 13' },
  { id: 'PLY-384756', channel: 'Influencer', ltv: '$1,203.88', segment: 'Dolphin', status: 'Active', date: 'Jan 12' },
  { id: 'PLY-475839', channel: 'App Store', ltv: '$567.90', segment: 'Minnow', status: 'At Risk', date: 'Jan 11' },
];

const segmentData = [
  { name: 'Whale', value: 2, color: '#22c55e' },
  { name: 'Dolphin', value: 8, color: '#4ade80' },
  { name: 'Minnow', value: 30, color: '#86efac' },
  { name: 'F2P', value: 60, color: '#1a1a1a' },
];

// Glassmorphism Card Component
const GlassCard = ({ children, className = '', glow = false }) => (
  <div 
    className={className}
    style={{
      background: 'linear-gradient(145deg, rgba(28, 28, 32, 0.9) 0%, rgba(18, 18, 22, 0.95) 100%)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderRadius: '20px',
      border: '1px solid rgba(255, 255, 255, 0.05)',
      boxShadow: glow 
        ? '0 8px 32px rgba(34, 197, 94, 0.15), inset 0 1px 0 rgba(255,255,255,0.05)'
        : '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
      position: 'relative',
      overflow: 'hidden',
    }}
  >
    {children}
  </div>
);

// Skeuomorphic Button
const SkeuButton = ({ children, variant = 'default', active = false, onClick }) => (
  <button
    onClick={onClick}
    style={{
      padding: '10px 18px',
      borderRadius: '12px',
      border: 'none',
      cursor: 'pointer',
      fontSize: '13px',
      fontWeight: 500,
      transition: 'all 0.2s ease',
      background: active 
        ? 'linear-gradient(145deg, #22c55e 0%, #16a34a 100%)'
        : variant === 'ghost'
          ? 'transparent'
          : 'linear-gradient(145deg, rgba(40, 40, 46, 0.9) 0%, rgba(28, 28, 32, 0.9) 100%)',
      color: active ? '#000' : '#fff',
      boxShadow: active
        ? '0 4px 15px rgba(34, 197, 94, 0.4), inset 0 1px 0 rgba(255,255,255,0.2)'
        : variant === 'ghost'
          ? 'none'
          : '0 4px 12px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05)',
    }}
  >
    {children}
  </button>
);

// Navigation Item
const NavItem = ({ icon, label, active = false, badge = null }) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: '14px',
      padding: '14px 18px',
      borderRadius: '14px',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      background: active 
        ? 'linear-gradient(145deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%)'
        : 'transparent',
      border: active ? '1px solid rgba(34, 197, 94, 0.2)' : '1px solid transparent',
      marginBottom: '4px',
    }}
  >
    <span style={{ fontSize: '18px', opacity: active ? 1 : 0.6 }}>{icon}</span>
    <span style={{ 
      fontSize: '14px', 
      fontWeight: active ? 600 : 400,
      color: active ? '#22c55e' : 'rgba(255,255,255,0.7)',
      flex: 1,
    }}>
      {label}
    </span>
    {badge && (
      <span style={{
        background: '#22c55e',
        color: '#000',
        fontSize: '11px',
        fontWeight: 600,
        padding: '2px 8px',
        borderRadius: '10px',
      }}>
        {badge}
      </span>
    )}
  </div>
);

// KPI Card with Skeuomorphic styling
const KPICard = ({ label, value, subtext, trend, icon }) => (
  <GlassCard>
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          {label}
        </span>
        <span style={{ fontSize: '20px', opacity: 0.4 }}>{icon}</span>
      </div>
      <div style={{ fontSize: '32px', fontWeight: 700, color: '#fff', marginBottom: '8px', fontFamily: 'SF Mono, monospace' }}>
        {value}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{
          fontSize: '12px',
          color: trend?.startsWith('+') ? '#22c55e' : trend?.startsWith('-') ? '#ef4444' : 'rgba(255,255,255,0.5)',
          background: trend?.startsWith('+') ? 'rgba(34, 197, 94, 0.1)' : trend?.startsWith('-') ? 'rgba(239, 68, 68, 0.1)' : 'rgba(255,255,255,0.05)',
          padding: '4px 10px',
          borderRadius: '8px',
          fontWeight: 500,
        }}>
          {trend}
        </span>
        <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.4)' }}>{subtext}</span>
      </div>
    </div>
  </GlassCard>
);

// Status Badge
const StatusBadge = ({ status }) => {
  const styles = {
    Active: { bg: 'rgba(34, 197, 94, 0.15)', color: '#22c55e', border: 'rgba(34, 197, 94, 0.3)' },
    Churned: { bg: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', border: 'rgba(239, 68, 68, 0.3)' },
    'At Risk': { bg: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b', border: 'rgba(245, 158, 11, 0.3)' },
  };
  const style = styles[status] || styles.Active;
  
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      padding: '6px 12px',
      borderRadius: '20px',
      fontSize: '12px',
      fontWeight: 500,
      background: style.bg,
      color: style.color,
      border: `1px solid ${style.border}`,
    }}>
      <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: style.color }} />
      {status}
    </span>
  );
};

// Main Dashboard Component
const LTVDashboard = () => {
  const [activeNav, setActiveNav] = useState('Dashboard');
  const [chartView, setChartView] = useState('line');
  
  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0c 0%, #111113 50%, #0d0d0f 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
      color: '#fff',
    }}>
      
      {/* Sidebar */}
      <aside style={{
        width: '280px',
        padding: '24px',
        borderRight: '1px solid rgba(255,255,255,0.05)',
        display: 'flex',
        flexDirection: 'column',
        background: 'linear-gradient(180deg, rgba(18, 18, 22, 0.8) 0%, rgba(12, 12, 14, 0.9) 100%)',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px', padding: '8px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px',
            boxShadow: '0 4px 15px rgba(34, 197, 94, 0.3)',
          }}>
            🎮
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '18px', letterSpacing: '-0.5px' }}>LTV Analytics</div>
            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>Marketing Intelligence</div>
          </div>
        </div>
        
        {/* Search */}
        <div style={{
          margin: '20px 0',
          padding: '12px 16px',
          borderRadius: '12px',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.05)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <span style={{ opacity: 0.4 }}>🔍</span>
          <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.3)' }}>Search anything...</span>
        </div>
        
        {/* Menu Section */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '1px', padding: '0 18px', marginBottom: '12px' }}>
            Menu
          </div>
          <NavItem icon="📊" label="Dashboard" active={activeNav === 'Dashboard'} />
          <NavItem icon="🔔" label="Notifications" badge="3" />
          <NavItem icon="📈" label="Analytics" />
          <NavItem icon="💳" label="Transactions" />
          <NavItem icon="🎴" label="Channels" />
          <NavItem icon="📜" label="History" />
        </div>
        
        {/* Features Section */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '1px', padding: '0 18px', marginBottom: '12px' }}>
            Features
          </div>
          <NavItem icon="🔗" label="Integration" />
          <NavItem icon="⚡" label="Automation" />
        </div>
        
        {/* Tools Section */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '1px', padding: '0 18px', marginBottom: '12px' }}>
            Tools
          </div>
          <NavItem icon="⚙️" label="Settings" />
          <NavItem icon="❓" label="Help Center" />
        </div>
        
        {/* Spacer */}
        <div style={{ flex: 1 }} />
        
        {/* Upgrade Card */}
        <GlassCard glow>
          <div style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
              <span style={{ fontSize: '24px' }}>✨</span>
              <span style={{ fontWeight: 600, color: '#22c55e' }}>Upgrade Pro!</span>
            </div>
            <p style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '16px', lineHeight: 1.5 }}>
              Get advanced LTV models and real-time predictions
            </p>
            <div style={{ display: 'flex', gap: '10px' }}>
              <SkeuButton active>Upgrade</SkeuButton>
              <SkeuButton variant="ghost">Learn more</SkeuButton>
            </div>
          </div>
        </GlassCard>
      </aside>
      
      {/* Main Content */}
      <main style={{ flex: 1, padding: '24px 32px', overflow: 'auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '28px' }}>
          <h1 style={{ fontSize: '28px', fontWeight: 700, letterSpacing: '-0.5px' }}>Dashboard</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <SkeuButton>📄 Generate Report</SkeuButton>
            <SkeuButton>↓ Export</SkeuButton>
            <div style={{
              padding: '10px 16px',
              borderRadius: '12px',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.05)',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '13px',
              color: 'rgba(255,255,255,0.5)',
            }}>
              🔍 Search Anything...
            </div>
          </div>
        </div>
        
        {/* KPI Row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '24px' }}>
          <KPICard 
            label="Total Predicted LTV" 
            value="$4,780,000" 
            trend="+12.3%" 
            subtext="vs last month"
            icon="💰"
          />
          <KPICard 
            label="Average Player LTV" 
            value="$478.00" 
            trend="+8.7%" 
            subtext="vs last month"
            icon="📊"
          />
          <KPICard 
            label="Model Accuracy (R²)" 
            value="82.5%" 
            trend="+2.1%" 
            subtext="vs baseline"
            icon="🎯"
          />
        </div>
        
        {/* Middle Row - 2 columns */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '20px', marginBottom: '24px' }}>
          {/* Channel Performance Card */}
          <GlassCard>
            <div style={{ padding: '24px' }}>
              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px' }}>Channel ROI Distribution</h3>
                <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.4)' }}>LTV allocation by source</p>
              </div>
              
              {/* Channel Bars */}
              <div style={{ marginBottom: '24px' }}>
                {channelData.slice(0, 4).map((channel, i) => (
                  <div key={i} style={{ marginBottom: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                      <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.7)' }}>{channel.name}</span>
                      <span style={{ fontSize: '13px', color: '#22c55e', fontWeight: 500 }}>{channel.roi}x ROI</span>
                    </div>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${Math.min(channel.roi / 2, 100)}%`,
                        background: 'linear-gradient(90deg, #22c55e 0%, #4ade80 100%)',
                        borderRadius: '4px',
                        boxShadow: '0 0 10px rgba(34, 197, 94, 0.5)',
                      }} />
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Credit Card Style Display */}
              <div style={{
                background: 'linear-gradient(135deg, #1a1a1e 0%, #0f0f12 100%)',
                borderRadius: '16px',
                padding: '20px',
                border: '1px solid rgba(255,255,255,0.08)',
                position: 'relative',
                overflow: 'hidden',
              }}>
                <div style={{
                  position: 'absolute',
                  top: '-20px',
                  right: '-20px',
                  width: '100px',
                  height: '100px',
                  background: 'radial-gradient(circle, rgba(34, 197, 94, 0.2) 0%, transparent 70%)',
                }} />
                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', marginBottom: '8px' }}>Top Performing Channel</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#22c55e', marginBottom: '16px' }}>Cross Promo</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>Players</div>
                    <div style={{ fontSize: '16px', fontWeight: 600 }}>961</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>LTV:CAC</div>
                    <div style={{ fontSize: '16px', fontWeight: 600, color: '#22c55e' }}>166x</div>
                  </div>
                </div>
              </div>
            </div>
          </GlassCard>
          
          {/* Revenue Chart */}
          <GlassCard>
            <div style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px' }}>LTV Trend Analysis</h3>
                  <div style={{ fontSize: '28px', fontWeight: 700, color: '#fff' }}>$4,780,000</div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <SkeuButton active={chartView === 'line'} onClick={() => setChartView('line')}>Line view</SkeuButton>
                  <SkeuButton active={chartView === 'bar'} onClick={() => setChartView('bar')}>Bar view</SkeuButton>
                </div>
              </div>
              
              <div style={{ height: '220px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  {chartView === 'line' ? (
                    <AreaChart data={monthlyData}>
                      <defs>
                        <linearGradient id="ltvGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
                          <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="month" stroke="rgba(255,255,255,0.3)" fontSize={12} />
                      <YAxis stroke="rgba(255,255,255,0.3)" fontSize={12} tickFormatter={v => `$${v}`} />
                      <Tooltip 
                        contentStyle={{ 
                          background: 'rgba(20,20,24,0.95)', 
                          border: '1px solid rgba(255,255,255,0.1)', 
                          borderRadius: '12px',
                          boxShadow: '0 8px 32px rgba(0,0,0,0.4)'
                        }}
                        formatter={(v) => [`$${v}`, 'Avg LTV']}
                      />
                      <Area type="monotone" dataKey="ltv" stroke="#22c55e" strokeWidth={3} fill="url(#ltvGradient)" />
                    </AreaChart>
                  ) : (
                    <BarChart data={monthlyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="month" stroke="rgba(255,255,255,0.3)" fontSize={12} />
                      <YAxis stroke="rgba(255,255,255,0.3)" fontSize={12} tickFormatter={v => `$${v}`} />
                      <Tooltip 
                        contentStyle={{ 
                          background: 'rgba(20,20,24,0.95)', 
                          border: '1px solid rgba(255,255,255,0.1)', 
                          borderRadius: '12px' 
                        }}
                      />
                      <Bar dataKey="ltv" fill="#22c55e" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  )}
                </ResponsiveContainer>
              </div>
              
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px', 
                marginTop: '16px',
                padding: '12px 16px',
                background: 'rgba(34, 197, 94, 0.08)',
                borderRadius: '10px',
                border: '1px solid rgba(34, 197, 94, 0.15)',
              }}>
                <span style={{ color: '#22c55e' }}>↑</span>
                <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.7)' }}>
                  <strong style={{ color: '#22c55e' }}>+12.3%</strong> vs last month
                </span>
              </div>
            </div>
          </GlassCard>
        </div>
        
        {/* Player Table */}
        <GlassCard>
          <div style={{ padding: '24px' }}>
            {/* Table Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 16px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.05)',
                width: '280px',
              }}>
                <span style={{ opacity: 0.4 }}>🔍</span>
                <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.4)' }}>Search Players...</span>
              </div>
              
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <SkeuButton>📅 Date Range</SkeuButton>
                <SkeuButton>🏷️ Segment</SkeuButton>
                <SkeuButton>More ▾</SkeuButton>
                <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)' }} />
                <SkeuButton>↓ Import</SkeuButton>
                <SkeuButton active>↑ Export</SkeuButton>
              </div>
            </div>
            
            {/* Table */}
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Player ID', 'Channel', 'Predicted LTV', 'Segment', 'Processed Date', 'Status'].map((header, i) => (
                    <th key={i} style={{
                      textAlign: 'left',
                      padding: '14px 16px',
                      fontSize: '11px',
                      fontWeight: 600,
                      color: 'rgba(255,255,255,0.4)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                    }}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {playerData.map((player, i) => (
                  <tr key={i} style={{ 
                    transition: 'background 0.2s',
                    cursor: 'pointer',
                  }}>
                    <td style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '14px', fontWeight: 500 }}>
                      {player.id}
                    </td>
                    <td style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '8px',
                          background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.05) 100%)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '14px',
                        }}>
                          {player.channel === 'Referral' ? '🤝' : player.channel === 'Paid Social' ? '📱' : player.channel === 'Organic' ? '🔍' : player.channel === 'Influencer' ? '⭐' : '📲'}
                        </div>
                        <span style={{ fontSize: '14px' }}>{player.channel}</span>
                      </div>
                    </td>
                    <td style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '14px', fontWeight: 600, color: '#22c55e' }}>
                      {player.ltv}
                    </td>
                    <td style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <span style={{
                        padding: '6px 12px',
                        borderRadius: '8px',
                        fontSize: '12px',
                        fontWeight: 500,
                        background: player.segment === 'Whale' 
                          ? 'rgba(34, 197, 94, 0.15)' 
                          : player.segment === 'Dolphin'
                            ? 'rgba(74, 222, 128, 0.15)'
                            : 'rgba(255,255,255,0.05)',
                        color: player.segment === 'Whale' || player.segment === 'Dolphin' ? '#22c55e' : 'rgba(255,255,255,0.6)',
                        border: `1px solid ${player.segment === 'Whale' || player.segment === 'Dolphin' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(255,255,255,0.05)'}`,
                      }}>
                        {player.segment}
                      </span>
                    </td>
                    <td style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '14px', color: 'rgba(255,255,255,0.6)' }}>
                      {player.date}
                    </td>
                    <td style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <StatusBadge status={player.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            
            {/* Pagination */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                {[1, 2, 3, '...', 20].map((page, i) => (
                  <button key={i} style={{
                    width: page === '...' ? 'auto' : '36px',
                    height: '36px',
                    borderRadius: '10px',
                    border: 'none',
                    background: page === 1 ? 'linear-gradient(145deg, #22c55e 0%, #16a34a 100%)' : 'rgba(255,255,255,0.03)',
                    color: page === 1 ? '#000' : 'rgba(255,255,255,0.6)',
                    fontSize: '13px',
                    fontWeight: page === 1 ? 600 : 400,
                    cursor: 'pointer',
                    padding: page === '...' ? '0 12px' : 0,
                  }}>
                    {page}
                  </button>
                ))}
              </div>
              <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.4)' }}>
                Showing 1 to 5 of 10,000 entries
                <button style={{
                  marginLeft: '12px',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                  background: 'transparent',
                  color: '#22c55e',
                  fontSize: '13px',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}>
                  Show All
                </button>
              </span>
            </div>
          </div>
        </GlassCard>
      </main>
    </div>
  );
};

export default LTVDashboard;
