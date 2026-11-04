/**
 * ThoraxAI — Clinical Pulmonary Tuberculosis Diagnostic Workstation
 * Architecture:
 * 1. Dual Radiograph Viewports (Input Monochrome CXR & Grad-CAM Heatmap)
 * 2. Persistent Scan History & Clinical Audit Trail (Local Storage backed)
 * 3. Dual X-Ray Embedded Printable Referral & Clearance Reports
 * 4. Regulatory Medical AI Disclaimer Banner
 * 5. 100% Mobile & Desktop Responsive Obsidian Dark Glass Theme
 */

import React, { useState, useEffect } from 'react';
import {
  Activity,
  Upload,
  Zap,
  RotateCcw,
  AlertOctagon,
  FileText,
  UserPlus,
  ShieldCheck,
  Cpu,
  RefreshCw,
  Sparkles,
  Layers,
  Printer,
  X,
  ChevronRight,
  Crosshair,
  FileCheck2,
  History,
  Trash2
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [diagnosticData, setDiagnosticData] = useState(null);
  const [engineOnline, setEngineOnline] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Persistent Scan History (Audit Trail)
  const [scanHistory, setScanHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('thoraxai_scan_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Referral Modal & Patient Metadata State
  const [showReferralModal, setShowReferralModal] = useState(false);
  const [caseId, setCaseId] = useState('');
  const [timestamp, setTimestamp] = useState('');
  const [patientMeta, setPatientMeta] = useState({
    patientName: '',
    age: '',
    gender: 'Male',
    patientId: '',
    referringDoctor: 'Dr. P. Mathur (Chief Medical Officer)',
    facility: 'District Pulmonology & NTEP Center'
  });

  // Engine health probe heartbeat
  useEffect(() => {
    let isMounted = true;
    const probeHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok && isMounted) {
          const payload = await res.json();
          setEngineOnline(Boolean(payload?.model_loaded));
        }
      } catch {
        if (isMounted) setEngineOnline(false);
      }
    };

    probeHealth();
    const interval = setInterval(probeHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const saveToHistory = (newEntry) => {
    setScanHistory((prev) => {
      const updated = [newEntry, ...prev.slice(0, 19)];
      try {
        localStorage.setItem('thoraxai_scan_history', JSON.stringify(updated));
      } catch (err) {
        console.warn('Storage quota exceeded', err);
      }
      return updated;
    });
  };

  const handleClearHistory = () => {
    localStorage.removeItem('thoraxai_scan_history');
    setScanHistory([]);
  };

  const handleLoadPastScan = (item) => {
    setCaseId(item.caseId);
    setTimestamp(item.timestamp);
    setSelectedFile({ name: item.fileName });
    setPreviewUrl(item.originalImage);
    setDiagnosticData(item.diagnosticData);
    setPatientMeta((prev) => ({
      ...prev,
      patientId: item.patientId || prev.patientId
    }));
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setDiagnosticData(null);
    setErrorMessage(null);
    setShowReferralModal(false);
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errText = 'Diagnostic inference execution failed.';
        try {
          const errPayload = await res.json();
          errText = errPayload?.detail || errText;
        } catch {}
        throw new Error(errText);
      }

      const result = await res.json();
      setDiagnosticData(result);

      const generatedCaseId = `TX-${Math.floor(100000 + Math.random() * 900000)}`;
      const generatedPtId = `UHID-${Math.floor(10000 + Math.random() * 90000)}`;
      const formattedTime = new Date().toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });

      setCaseId(generatedCaseId);
      setTimestamp(formattedTime);
      setPatientMeta((prev) => ({
        ...prev,
        patientId: prev.patientId || generatedPtId,
        facility: result?.diagnosis?.is_positive
          ? 'District Pulmonology & NTEP DOTS Center'
          : 'General Outpatient Department (OPD Clearance)'
      }));

      // Persist in local storage audit log
      saveToHistory({
        caseId: generatedCaseId,
        patientId: generatedPtId,
        fileName: selectedFile.name,
        timestamp: formattedTime,
        isPositive: Boolean(result?.diagnosis?.is_positive),
        label: result?.diagnosis?.label,
        confidence: result?.diagnosis?.confidence_score,
        originalImage: result?.visuals?.original_image || previewUrl,
        gradcamImage: result?.visuals?.gradcam_heatmap,
        diagnosticData: result
      });
    } catch (err) {
      setErrorMessage(err?.message || 'Server connection error.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleResetScan = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setDiagnosticData(null);
    setErrorMessage(null);
    setShowReferralModal(false);
  };

  const handleMetaChange = (e) => {
    const { name, value } = e.target;
    setPatientMeta((prev) => ({ ...prev, [name]: value }));
  };

  const isPositive = Boolean(diagnosticData?.diagnosis?.is_positive);
  const visuals = diagnosticData?.visuals || {};
  const diagnosis = diagnosticData?.diagnosis || {};
  const triage = diagnosticData?.triage || {};
  const quality = diagnosticData?.quality_gate || {};

  return (
    <div className="min-h-screen bg-[#06080D] text-slate-100 flex flex-col font-sans selection:bg-[#FF4B32]/30 selection:text-white relative overflow-x-hidden">
      {/* Background Cyber Grid & Subtle Radial Accents */}
      <div
        className="pointer-events-none fixed inset-0 opacity-20"
        style={{
          backgroundImage: 'radial-gradient(rgba(255, 255, 255, 0.12) 1px, transparent 1px)',
          backgroundSize: '28px 28px'
        }}
      />
      <div className="pointer-events-none fixed top-[-15%] left-[10%] w-[500px] h-[500px] rounded-full bg-[#FF4B32]/10 blur-[140px]" />
      <div className="pointer-events-none fixed bottom-[-15%] right-[10%] w-[500px] h-[500px] rounded-full bg-cyan-500/10 blur-[140px]" />

      {/* Top Telemetry Bar */}
      <div className="border-b border-white/[0.08] bg-[#0A0D14]/80 backdrop-blur-xl px-4 sm:px-6 py-2.5 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-[11px] sm:text-xs font-mono">
          <div className="flex items-center gap-2 text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-[#FF4B32] shrink-0" />
            <span className="truncate">Core: DenseNet-121 • PyTorch</span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                engineOnline ? 'bg-emerald-400 animate-pulse shadow-[0_0_10px_#34d399]' : 'bg-rose-500'
              }`}
            />
            <span className={`font-bold ${engineOnline ? 'text-emerald-400' : 'text-rose-400'}`}>
              {engineOnline ? 'ENGINE ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-10 flex flex-col gap-6 sm:gap-8 z-10">
        {/* 1. Center Hero Header */}
        <header className="text-center flex flex-col items-center justify-center space-y-2.5 px-2">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.1] text-slate-300 text-[10px] sm:text-xs font-mono tracking-wider">
            <Activity className="w-3.5 h-3.5 text-[#FF4B32]" />
            <span>AI PULMONARY DIAGNOSTIC &amp; SALIENCY WORKSTATION</span>
          </div>

          <h1 className="text-4xl sm:text-6xl md:text-7xl font-black tracking-tight text-white uppercase leading-none">
            THORAX <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#FF4B32] to-amber-500">AI</span>
          </h1>

          <p className="text-xs sm:text-base text-slate-400 max-w-2xl font-normal leading-relaxed">
            Automated anteroposterior chest radiograph screening, out-of-distribution quality verification, and Grad-CAM spatial localization.
          </p>
        </header>

        {/* 2. Side-by-Side Dual Viewports */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
          {/* Viewport 1: Input Radiograph */}
          <div className="bg-[#0F131D]/90 backdrop-blur-xl border border-white/[0.09] hover:border-white/[0.16] transition rounded-2xl p-4 sm:p-5 flex flex-col justify-between shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.07] mb-3 gap-2">
              <span className="text-xs sm:text-sm font-bold tracking-wider uppercase text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-[#FF4B32] shrink-0" />
                <span>Input Radiograph (CXR)</span>
              </span>
              {selectedFile ? (
                <span className="font-mono text-[10px] sm:text-xs text-amber-300 truncate max-w-[140px] sm:max-w-[200px] bg-amber-500/10 px-2.5 py-0.5 rounded border border-amber-500/20">
                  {selectedFile.name}
                </span>
              ) : (
                <span className="text-[10px] sm:text-xs font-mono text-slate-500">AWAITING INPUT</span>
              )}
            </div>

            <div className="relative w-full h-[280px] sm:h-[380px] bg-[#040508] rounded-xl border border-white/[0.07] overflow-hidden flex items-center justify-center shadow-inner">
              <div className="pointer-events-none absolute top-3 left-3 font-mono text-[10px] sm:text-xs font-bold text-white/30 border border-white/10 px-2 py-0.5 rounded">
                R
              </div>
              <div className="pointer-events-none absolute top-3 right-3 font-mono text-[10px] sm:text-xs font-bold text-white/30 border border-white/10 px-2 py-0.5 rounded">
                L
              </div>

              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Raw CXR Preview"
                  className="max-h-full max-w-full object-contain select-none p-2"
                />
              ) : (
                <label className="w-full h-full flex flex-col items-center justify-center cursor-pointer p-4 sm:p-6 text-center hover:bg-white/[0.02] transition group">
                  <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-gradient-to-br from-[#FF4B32]/20 to-amber-500/5 border border-[#FF4B32]/40 flex items-center justify-center text-[#FF4B32] mb-3 shadow-[0_0_30px_-5px_rgba(255,75,50,0.3)] group-hover:scale-105 transition-transform">
                    <Upload className="w-6 h-6 sm:w-7 sm:h-7" />
                  </div>
                  <span className="text-sm sm:text-base font-bold text-white">Upload Chest Radiograph</span>
                  <p className="text-[11px] sm:text-xs text-slate-400 mt-1">
                    Tap to browse or drag &amp; drop PNG, JPG, or DICOM
                  </p>
                  <span className="mt-4 px-4 py-2 rounded-xl bg-white/[0.06] border border-white/[0.12] text-xs font-semibold text-white group-hover:bg-[#FF4B32] group-hover:border-[#FF4B32] transition">
                    Select Radiograph File
                  </span>
                  <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                </label>
              )}
            </div>
          </div>

          {/* Viewport 2: Grad-CAM Saliency Heatmap */}
          <div className="bg-[#0F131D]/90 backdrop-blur-xl border border-white/[0.09] hover:border-white/[0.16] transition rounded-2xl p-4 sm:p-5 flex flex-col justify-between shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.07] mb-3 gap-2">
              <span className="text-xs sm:text-sm font-bold tracking-wider uppercase text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-[#FF4B32] shrink-0" />
                <span>Grad-CAM Saliency Map</span>
              </span>
              <span
                className={`text-[10px] sm:text-xs font-mono font-bold px-2.5 py-0.5 rounded border ${
                  diagnosticData
                    ? isPositive
                      ? 'bg-[#FF4B32]/20 text-[#FF4B32] border-[#FF4B32]/40'
                      : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                    : 'bg-white/[0.04] text-slate-500 border-white/[0.08]'
                }`}
              >
                {diagnosticData
                  ? isPositive
                    ? 'HOTSPOT LOCALIZED'
                    : 'CLEAR LUNG FIELDS'
                  : 'STANDBY'}
              </span>
            </div>

            <div className="relative w-full h-[280px] sm:h-[380px] bg-[#040508] rounded-xl border border-white/[0.07] overflow-hidden flex items-center justify-center shadow-inner">
              <div className="pointer-events-none absolute top-3 left-3 font-mono text-[10px] sm:text-xs font-bold text-white/30 border border-white/10 px-2 py-0.5 rounded">
                R
              </div>
              <div className="pointer-events-none absolute top-3 right-3 font-mono text-[10px] sm:text-xs font-bold text-white/30 border border-white/10 px-2 py-0.5 rounded">
                L
              </div>

              {visuals?.gradcam_heatmap ? (
                <>
                  <img
                    src={visuals.gradcam_heatmap}
                    alt="Grad-CAM Saliency Overlay"
                    className="max-h-full max-w-full object-contain select-none p-2"
                  />
                  <div className="pointer-events-none absolute bottom-3 right-3 font-mono text-[9px] sm:text-[10px] text-slate-300 bg-black/80 px-2.5 py-1 rounded border border-white/10 backdrop-blur-md">
                    LAYER: DENSENET.FEATURES[-1]
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-6 text-slate-500">
                  <Crosshair className="w-10 h-10 sm:w-12 sm:h-12 mb-3 stroke-[1.2] text-slate-600 animate-pulse" />
                  <span className="text-xs sm:text-sm font-bold text-slate-300 uppercase tracking-wider">
                    Saliency Map Offline
                  </span>
                  <p className="text-[11px] sm:text-xs text-slate-500 mt-1 max-w-[240px]">
                    Execute diagnostic inference to project spatial heatmaps over pulmonary zones.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* 3. Action Buttons Section */}
        <section className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 sm:gap-4">
          <button
            type="button"
            onClick={handleResetScan}
            disabled={(!previewUrl && !diagnosticData) || isAnalyzing}
            className="px-6 py-3.5 sm:py-4 rounded-xl bg-[#131824] hover:bg-[#1C2333] border border-white/[0.1] text-slate-200 text-xs sm:text-sm font-bold uppercase tracking-wider flex items-center justify-center gap-2.5 transition cursor-pointer disabled:opacity-35 disabled:pointer-events-none"
          >
            <RotateCcw className="w-4 h-4 text-slate-400" />
            <span>New Scan (Clear)</span>
          </button>

          <button
            type="button"
            disabled={!selectedFile || isAnalyzing}
            onClick={handleRunInference}
            className="flex-1 py-4 px-6 rounded-xl bg-gradient-to-r from-[#FF4B32] to-[#FF6B35] hover:from-[#e83e26] hover:to-[#f05d28] text-white font-black text-xs sm:text-sm uppercase tracking-widest flex items-center justify-center gap-2.5 shadow-[0_0_35px_-5px_rgba(255,75,50,0.45)] transition disabled:opacity-35 disabled:pointer-events-none cursor-pointer"
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="w-4 h-4 sm:w-5 sm:h-5 animate-spin text-white" />
                <span>Executing Neural Diagnostic Pipeline...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 sm:w-5 sm:h-5 fill-current" />
                <span>Run Diagnostic Inference Check</span>
              </>
            )}
          </button>
        </section>

        {/* Error Notification Banner */}
        {errorMessage && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs sm:text-sm flex items-start sm:items-center gap-3 shadow-lg">
            <AlertOctagon className="w-5 h-5 text-rose-400 shrink-0 mt-0.5 sm:mt-0" />
            <span className="font-medium leading-relaxed">{errorMessage}</span>
          </div>
        )}

        {/* 4. Diagnostic Verdict Card */}
        {diagnosticData ? (
          <section className="flex flex-col gap-5">
            <div
              className={`p-5 sm:p-8 rounded-2xl border shadow-2xl relative overflow-hidden backdrop-blur-xl ${
                isPositive
                  ? 'bg-gradient-to-br from-[#FF4B32]/15 via-[#140D10] to-[#0F131D] border-[#FF4B32]/45 shadow-[0_0_60px_-15px_rgba(255,75,50,0.25)]'
                  : 'bg-gradient-to-br from-emerald-500/15 via-[#0B1612] to-[#0F131D] border-emerald-500/40 shadow-[0_0_60px_-15px_rgba(16,185,129,0.2)]'
              }`}
            >
              {/* Finding Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/[0.09] mb-5">
                <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                  <span className="font-mono text-[11px] sm:text-xs font-bold tracking-widest uppercase text-slate-400">
                    DIAGNOSTIC VERDICT
                  </span>
                  <span className="font-mono text-[11px] sm:text-xs px-2.5 py-0.5 rounded bg-black/50 border border-white/10 text-white font-bold">
                    {caseId}
                  </span>
                </div>

                <span
                  className={`font-mono text-[10px] sm:text-xs font-black px-3.5 py-1 rounded-full border ${
                    isPositive
                      ? 'bg-[#FF4B32]/20 text-[#FF4B32] border-[#FF4B32]/40 animate-pulse'
                      : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  }`}
                >
                  {triage?.urgency || (isPositive ? 'TIER 1: HIGH PRIORITY URGENT' : 'TIER 3: ROUTINE CLEAR')}
                </span>
              </div>

              {/* Giant Headline & Meters */}
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                <div>
                  <span className="text-[11px] sm:text-xs uppercase font-mono tracking-wider text-slate-400 block mb-1">
                    Primary Radiographic Finding
                  </span>
                  <h2
                    className={`text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight uppercase leading-none ${
                      isPositive ? 'text-[#FF4B32]' : 'text-emerald-400'
                    }`}
                  >
                    {diagnosis?.label}
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-300 mt-2.5 font-mono">
                    Model Posterior Confidence:{' '}
                    <span className="font-bold text-white text-sm sm:text-base">
                      {diagnosis?.confidence_score}%
                    </span>
                  </p>
                </div>

                {/* Probability Bars */}
                <div className="w-full lg:w-96 space-y-3.5 font-mono text-xs bg-black/50 p-4 sm:p-5 rounded-xl border border-white/[0.08]">
                  <div>
                    <div className="flex justify-between mb-1.5 font-bold">
                      <span className="text-slate-300">Tuberculosis Probability</span>
                      <span className="text-[#FF4B32]">{diagnosis?.tb_probability}%</span>
                    </div>
                    <div className="w-full h-2.5 rounded-full bg-slate-900 overflow-hidden p-0.5 border border-white/5">
                      <div
                        className="h-full bg-gradient-to-r from-amber-500 to-[#FF4B32] rounded-full transition-all duration-700"
                        style={{ width: `${diagnosis?.tb_probability}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5 font-semibold">
                      <span className="text-slate-400">Normal / Infiltration-Free</span>
                      <span className="text-emerald-400">{diagnosis?.normal_probability}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full transition-all duration-700"
                        style={{ width: `${diagnosis?.normal_probability}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Dynamic Clinical Protocol Checklist */}
              <div className="mt-6 pt-5 border-t border-white/[0.09]">
                <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2 mb-3">
                  <FileText className="w-4 h-4 text-[#FF4B32]" />
                  <span>Clinical Protocol Advisory &amp; Action Items</span>
                </span>

                <div className="bg-black/40 p-4 sm:p-5 rounded-xl border border-white/[0.08] space-y-3">
                  <div className="text-xs sm:text-sm text-white font-semibold pb-2.5 border-b border-white/[0.07] flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${isPositive ? 'bg-[#FF4B32]' : 'bg-emerald-400'}`} />
                    <span>{triage?.recommendation}</span>
                  </div>

                  {isPositive ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
                      <div className="flex items-start gap-2.5 bg-white/[0.03] p-3 rounded-lg border border-white/[0.06]">
                        <span className="font-mono text-[#FF4B32] font-bold">01.</span>
                        <div>
                          <strong className="block text-white font-semibold">Sputum GeneXpert MTB/RIF</strong>
                          <span className="text-slate-400 text-[11px]">Dispatch specimen for rapid molecular confirmation &amp; Rifampicin resistance.</span>
                        </div>
                      </div>

                      <div className="flex items-start gap-2.5 bg-white/[0.03] p-3 rounded-lg border border-white/[0.06]">
                        <span className="font-mono text-[#FF4B32] font-bold">02.</span>
                        <div>
                          <strong className="block text-white font-semibold">AFB Smear Microscopy</strong>
                          <span className="text-slate-400 text-[11px]">Collect spot &amp; early-morning sputum samples for Acid-Fast Bacilli staining.</span>
                        </div>
                      </div>

                      <div className="flex items-start gap-2.5 bg-white/[0.03] p-3 rounded-lg border border-white/[0.06]">
                        <span className="font-mono text-[#FF4B32] font-bold">03.</span>
                        <div>
                          <strong className="block text-white font-semibold">Airborne Isolation Protocol</strong>
                          <span className="text-slate-400 text-[11px]">Initiate respiratory precautions until bacteriologic status is confirmed.</span>
                        </div>
                      </div>

                      <div className="flex items-start gap-2.5 bg-white/[0.03] p-3 rounded-lg border border-white/[0.06]">
                        <span className="font-mono text-[#FF4B32] font-bold">04.</span>
                        <div>
                          <strong className="block text-white font-semibold">NTEP / DOTS Registration</strong>
                          <span className="text-slate-400 text-[11px]">Generate clinical referral slip and route patient to pulmonary OPD.</span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
                      <div className="flex items-start gap-2.5 bg-white/[0.03] p-3 rounded-lg border border-white/[0.06]">
                        <span className="font-mono text-emerald-400 font-bold">01.</span>
                        <div>
                          <strong className="block text-white font-semibold">No Active Parenchymal Consolidation</strong>
                          <span className="text-slate-400 text-[11px]">Lung fields appear radiologically clear of focal cavitary or apical lesions.</span>
                        </div>
                      </div>
                      <div className="flex items-start gap-2.5 bg-white/[0.03] p-3 rounded-lg border border-white/[0.06]">
                        <span className="font-mono text-emerald-400 font-bold">02.</span>
                        <div>
                          <strong className="block text-white font-semibold">Clinical Correlation Advised</strong>
                          <span className="text-slate-400 text-[11px]">If persistent cough (&gt;2 weeks) or fever exists, consider standard sputum evaluation.</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Quality Gate Telemetry Bar */}
              <div className="mt-5 flex flex-wrap items-center gap-3 sm:gap-4 text-[11px] sm:text-xs font-mono text-slate-400 bg-black/40 p-3 rounded-xl border border-white/[0.06]">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Quality Gate: <strong className="text-emerald-400">PASSED</strong></span>
                </div>
                <span className="hidden sm:inline">•</span>
                <span>Sharpness: <strong className="text-white">{quality?.sharpness_score}</strong></span>
                <span className="hidden sm:inline">•</span>
                <span>Contrast: <strong className="text-white">{quality?.contrast_score}</strong></span>
              </div>

              {/* Regulatory AI Disclaimer Banner */}
              <div className="mt-4 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-2.5 font-mono">
                <AlertOctagon className="w-4 h-4 text-amber-400 shrink-0" />
                <span>
                  <strong>Clinical Disclaimer:</strong> This system provides AI-based probabilistic screening predictions and may occasionally misclassify inputs. Final diagnostic validation must be certified by a medical professional via confirmatory sputum testing.
                </span>
              </div>
            </div>

            {/* 5. Universal Report & Referral Trigger */}
            <div
              className={`p-5 sm:p-6 rounded-2xl border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-2xl ${
                isPositive
                  ? 'bg-gradient-to-r from-[#FF4B32]/20 via-[#161218] to-[#0F131D] border-[#FF4B32]/45'
                  : 'bg-gradient-to-r from-emerald-500/15 via-[#0E1916] to-[#0F131D] border-emerald-500/40'
              }`}
            >
              <div className="flex items-start sm:items-center gap-4">
                <div
                  className={`w-12 h-12 rounded-2xl text-white flex items-center justify-center shrink-0 shadow-lg ${
                    isPositive ? 'bg-[#FF4B32] shadow-[#FF4B32]/30' : 'bg-emerald-600 shadow-emerald-600/30'
                  }`}
                >
                  {isPositive ? <UserPlus className="w-6 h-6" /> : <FileCheck2 className="w-6 h-6" />}
                </div>
                <div>
                  <h4 className="text-sm sm:text-base font-extrabold text-white">
                    {isPositive
                      ? 'Action Required: Official Patient Referral & GeneXpert Dispatch'
                      : 'Diagnostic Clearance: Issue Official Radiological Screening Report'}
                  </h4>
                  <p className="text-xs text-slate-300 mt-0.5">
                    {isPositive
                      ? 'Generate a printable NTEP/DOTS referral slip with embedded dual CXR & Grad-CAM visual evidence.'
                      : 'Generate a printable screening clearance certificate with embedded CXR radiographs for patient records.'}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setShowReferralModal(true)}
                className={`w-full sm:w-auto px-6 py-3.5 rounded-xl text-white font-black text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg transition cursor-pointer shrink-0 ${
                  isPositive
                    ? 'bg-[#FF4B32] hover:bg-[#e03a22] shadow-[#FF4B32]/30'
                    : 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-600/30'
                }`}
              >
                <Printer className="w-4 h-4" />
                <span>{isPositive ? 'Generate Referral Slip' : 'Generate Screening Report'}</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </section>
        ) : (
          /* Standby Card */
          <div className="p-8 sm:p-10 rounded-2xl bg-[#0F131D]/80 border border-white/[0.08] flex flex-col items-center justify-center text-center">
            <Activity className="w-10 h-10 sm:w-12 sm:h-12 text-slate-600 mb-3" />
            <h3 className="text-xs sm:text-sm font-bold text-slate-300 uppercase tracking-wider">
              Clinical Telemetry Standby
            </h3>
            <p className="text-xs text-slate-500 mt-1 max-w-sm">
              Upload a chest radiograph and run diagnostic inference to populate neural probabilities, Grad-CAM heatmaps, and printable clinical slips.
            </p>
          </div>
        )}

        {/* 6. Persistent Scan History & Clinical Audit Trail */}
        <section className="bg-[#0F131D]/90 backdrop-blur-xl border border-white/[0.09] rounded-2xl p-5 sm:p-6 shadow-2xl">
          <div className="flex items-center justify-between pb-4 border-b border-white/[0.07] mb-4">
            <div className="flex items-center gap-2.5">
              <History className="w-5 h-5 text-[#FF4B32]" />
              <div>
                <h3 className="text-sm sm:text-base font-bold text-white uppercase tracking-wider">
                  Session Scan History &amp; Audit Trail
                </h3>
                <p className="text-[11px] text-slate-400 font-mono">
                  Persists across browser refreshes • {scanHistory.length} Record(s) Stored
                </p>
              </div>
            </div>

            {scanHistory.length > 0 && (
              <button
                type="button"
                onClick={handleClearHistory}
                className="px-3 py-1.5 rounded-lg bg-white/[0.04] hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 text-xs font-mono flex items-center gap-1.5 transition cursor-pointer border border-white/[0.08]"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear Audit</span>
              </button>
            )}
          </div>

          {scanHistory.length === 0 ? (
            <div className="text-center py-8 text-xs font-mono text-slate-500">
              No scans archived yet. Complete an inference check to record audit entries.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-white/[0.06] text-slate-400 text-[10px] uppercase tracking-wider">
                    <th className="pb-3 font-semibold">Case ID</th>
                    <th className="pb-3 font-semibold">Timestamp</th>
                    <th className="pb-3 font-semibold">Specimen File</th>
                    <th className="pb-3 font-semibold">AI Finding</th>
                    <th className="pb-3 font-semibold">Confidence</th>
                    <th className="pb-3 text-right font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {scanHistory.map((item) => (
                    <tr key={item.caseId} className="hover:bg-white/[0.02] transition">
                      <td className="py-3 font-bold text-white">{item.caseId}</td>
                      <td className="py-3 text-slate-400 text-[11px]">{item.timestamp}</td>
                      <td className="py-3 text-slate-300 max-w-[140px] truncate">{item.fileName}</td>
                      <td className="py-3">
                        <span
                          className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${
                            item.isPositive
                              ? 'bg-[#FF4B32]/15 text-[#FF4B32] border-[#FF4B32]/30'
                              : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                          }`}
                        >
                          {item.label}
                        </span>
                      </td>
                      <td className="py-3 text-white font-bold">{item.confidence}%</td>
                      <td className="py-3 text-right">
                        <button
                          type="button"
                          onClick={() => handleLoadPastScan(item)}
                          className="px-3 py-1 rounded-lg bg-white/[0.06] hover:bg-[#FF4B32] text-slate-200 hover:text-white transition cursor-pointer text-[11px] font-sans font-semibold"
                        >
                          Reopen Scan
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      {/* 7. Complete Interactive & Printable Report Modal (Includes Embedded Dual X-Rays) */}
      {showReferralModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/85 backdrop-blur-md overflow-y-auto">
          <div className="bg-[#111520] border border-white/[0.15] w-full max-w-4xl rounded-2xl p-4 sm:p-7 shadow-2xl relative text-slate-100 my-auto max-h-[94vh] overflow-y-auto">
            {/* Modal Top Controls (Hidden when printing) */}
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.1] print:hidden">
              <div className="flex items-center gap-2.5">
                <div
                  className={`w-9 h-9 rounded-xl flex items-center justify-center font-black border ${
                    isPositive
                      ? 'bg-[#FF4B32]/20 border-[#FF4B32]/40 text-[#FF4B32]'
                      : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                  }`}
                >
                  TX
                </div>
                <div>
                  <h3 className="text-sm sm:text-base font-extrabold text-white uppercase tracking-wider">
                    {isPositive ? 'Clinical Referral Requisition Slip' : 'Radiological Screening Clearance Report'}
                  </h3>
                  <p className="text-[11px] text-slate-400 font-mono">
                    Enter patient details below — live preview updates automatically
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowReferralModal(false)}
                className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Interactive Metadata Form (Hidden during print) */}
            <div className="my-5 grid grid-cols-1 sm:grid-cols-3 gap-3 bg-black/40 p-4 rounded-xl border border-white/[0.08] print:hidden">
              <div className="sm:col-span-2">
                <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Patient Full Name *</label>
                <input
                  type="text"
                  name="patientName"
                  value={patientMeta.patientName}
                  onChange={handleMetaChange}
                  placeholder="e.g., Rajesh Kumar"
                  className="w-full px-3 py-2 rounded-lg bg-[#0A0D14] border border-white/[0.12] text-xs text-white focus:outline-none focus:border-[#FF4B32]"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Age *</label>
                  <input
                    type="text"
                    name="age"
                    value={patientMeta.age}
                    onChange={handleMetaChange}
                    placeholder="42 Yrs"
                    className="w-full px-3 py-2 rounded-lg bg-[#0A0D14] border border-white/[0.12] text-xs text-white focus:outline-none focus:border-[#FF4B32]"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Sex</label>
                  <select
                    name="gender"
                    value={patientMeta.gender}
                    onChange={handleMetaChange}
                    className="w-full px-2 py-2 rounded-lg bg-[#0A0D14] border border-white/[0.12] text-xs text-white focus:outline-none focus:border-[#FF4B32]"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Patient UHID / Reg No.</label>
                <input
                  type="text"
                  name="patientId"
                  value={patientMeta.patientId}
                  onChange={handleMetaChange}
                  className="w-full px-3 py-2 rounded-lg bg-[#0A0D14] border border-white/[0.12] text-xs font-mono text-white focus:outline-none focus:border-[#FF4B32]"
                />
              </div>

              <div>
                <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Issuing / Referring Officer</label>
                <input
                  type="text"
                  name="referringDoctor"
                  value={patientMeta.referringDoctor}
                  onChange={handleMetaChange}
                  className="w-full px-3 py-2 rounded-lg bg-[#0A0D14] border border-white/[0.12] text-xs text-white focus:outline-none focus:border-[#FF4B32]"
                />
              </div>

              <div>
                <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Target Department / Unit</label>
                <input
                  type="text"
                  name="facility"
                  value={patientMeta.facility}
                  onChange={handleMetaChange}
                  className="w-full px-3 py-2 rounded-lg bg-[#0A0D14] border border-white/[0.12] text-xs text-white focus:outline-none focus:border-[#FF4B32]"
                />
              </div>
            </div>

            {/* OFFICIAL PRINTABLE SLIP */}
            <div
              id="printable-clinical-report"
              className="bg-white text-slate-900 p-5 sm:p-7 rounded-xl border border-slate-300 shadow-inner font-sans space-y-4"
            >
              {/* Document Header */}
              <div className="flex flex-wrap items-start justify-between gap-4 pb-3 border-b-2 border-slate-900">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-black text-lg sm:text-xl tracking-tight text-slate-900 uppercase">
                      THORAX<span className="text-[#FF4B32]">AI</span> RADIOLOGY &amp; PULMONARY TRIAGE
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-600 font-semibold">
                    {isPositive
                      ? 'Official NTEP / DOTS Tuberculosis Referral & Confirmatory Requisition'
                      : 'Automated Chest Radiograph (CXR) Screening & Clearance Certificate'}
                  </p>
                </div>
                <div className="text-right font-mono text-[11px]">
                  <div className="font-bold text-slate-900">CASE ID: {caseId}</div>
                  <div className="text-slate-500">ISSUED: {timestamp}</div>
                </div>
              </div>

              {/* Patient & Officer Metadata Table */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 bg-slate-100 p-3.5 rounded-lg border border-slate-200 text-xs">
                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">PATIENT NAME:</span>
                  <strong className="text-slate-900 font-bold">{patientMeta.patientName || '________________________'}</strong>
                </div>
                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">AGE / SEX:</span>
                  <strong className="text-slate-900 font-bold">
                    {patientMeta.age ? `${patientMeta.age} / ${patientMeta.gender}` : `____ / ${patientMeta.gender}`}
                  </strong>
                </div>
                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">PATIENT UHID:</span>
                  <strong className="text-slate-900 font-mono font-bold">{patientMeta.patientId}</strong>
                </div>
                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">ISSUING OFFICER:</span>
                  <strong className="text-slate-900 font-semibold">{patientMeta.referringDoctor}</strong>
                </div>
                <div className="sm:col-span-2">
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">DESTINATION UNIT:</span>
                  <strong className="text-slate-900 font-semibold">{patientMeta.facility}</strong>
                </div>
              </div>

              {/* Dual Embedded X-Rays */}
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-500 font-bold block mb-2">
                  ATTACHED RADIOGRAPHIC VISUAL EVIDENCE (ANTEROPOSTERIOR CXR &amp; GRAD-CAM MAP):
                </span>
                <div className="grid grid-cols-2 gap-3">
                  <div className="border border-slate-300 rounded-lg p-2 bg-slate-950 flex flex-col items-center">
                    <div className="w-full flex justify-between items-center text-[9px] font-mono text-slate-300 pb-1.5 mb-1.5 border-b border-slate-800">
                      <span>FIG 1: INPUT MONOCHROME CXR</span>
                      <span>R / L</span>
                    </div>
                    <img
                      src={visuals?.original_image || previewUrl}
                      alt="Original CXR"
                      className="h-44 sm:h-52 object-contain"
                    />
                  </div>

                  <div className="border border-slate-300 rounded-lg p-2 bg-slate-950 flex flex-col items-center">
                    <div className="w-full flex justify-between items-center text-[9px] font-mono text-slate-300 pb-1.5 mb-1.5 border-b border-slate-800">
                      <span>FIG 2: GRAD-CAM SALIENCY MAP</span>
                      <span>{isPositive ? 'HOTSPOT ACTIVE' : 'CLEAR'}</span>
                    </div>
                    <img
                      src={visuals?.gradcam_heatmap || previewUrl}
                      alt="Grad-CAM CXR"
                      className="h-44 sm:h-52 object-contain"
                    />
                  </div>
                </div>
              </div>

              {/* AI Diagnostic Summary Box */}
              <div
                className={`p-3.5 rounded-lg border flex flex-wrap items-center justify-between gap-2 text-xs ${
                  isPositive
                    ? 'bg-rose-50 border-rose-300 text-rose-950'
                    : 'bg-emerald-50 border-emerald-300 text-emerald-950'
                }`}
              >
                <div>
                  <span className="text-[10px] font-mono uppercase font-bold block opacity-75">
                    DENSENET-121 RADIOLOGICAL SCREENING VERDICT:
                  </span>
                  <span className="text-sm font-black uppercase">
                    {diagnosis?.label} — {triage?.urgency || (isPositive ? 'TIER 1: URGENT' : 'ROUTINE CLEAR')}
                  </span>
                </div>
                <div className="font-mono text-right">
                  <span className="text-[10px] block opacity-75">CONFIDENCE &amp; QUALITY</span>
                  <strong className="text-sm font-black">
                    {diagnosis?.confidence_score}% (Sharpness: {quality?.sharpness_score})
                  </strong>
                </div>
              </div>

              {/* Clinical Orders / Clearance Notes */}
              <div className="text-xs space-y-1.5">
                <span className="text-[10px] font-mono uppercase text-slate-500 font-bold block">
                  {isPositive
                    ? 'MANDATORY CONFIRMATORY INVESTIGATIONS ORDERED:'
                    : 'CLINICAL SCREENING & FOLLOW-UP ADVISORY:'}
                </span>
                {isPositive ? (
                  <ul className="list-disc list-inside space-y-1 text-slate-800 font-medium">
                    <li>Cartridge-Based Nucleic Acid Amplification Test (CBNAAT / GeneXpert MTB/RIF)</li>
                    <li>Direct Sputum Smear Microscopy for Acid-Fast Bacilli (Spot &amp; Early Morning)</li>
                    <li>Immediate Pulmonology OPD Evaluation &amp; Airborne Isolation Precautions</li>
                  </ul>
                ) : (
                  <ul className="list-disc list-inside space-y-1 text-slate-800 font-medium">
                    <li>No active parenchymal consolidation or cavitary lesions detected on current CXR.</li>
                    <li>Radiologically cleared for routine outpatient record. Correlate clinically if symptoms persist.</li>
                  </ul>
                )}
              </div>

              {/* Signature & Stamp Footer */}
              <div className="pt-5 mt-3 border-t border-slate-300 flex items-end justify-between text-[11px] text-slate-600">
                <div className="max-w-xs text-[10px] italic text-slate-500">
                  * Computer-assisted DenseNet-121 screening report. To be correlated clinically by the attending Physician/Radiologist. AI-based prediction is probabilistic and subject to diagnostic error.
                </div>
                <div className="text-right">
                  <div className="w-40 border-b border-slate-400 mb-1 ml-auto" />
                  <span className="font-bold text-slate-800 block">{patientMeta.referringDoctor}</span>
                  <span className="text-[10px] text-slate-500">Authorized Signature &amp; Stamp</span>
                </div>
              </div>
            </div>

            {/* Modal Bottom Actions */}
            <div className="flex flex-wrap items-center justify-end gap-3 pt-4 mt-4 border-t border-white/[0.1] print:hidden">
              <button
                type="button"
                onClick={() => setShowReferralModal(false)}
                className="px-4 py-2.5 rounded-xl bg-white/[0.06] hover:bg-white/[0.12] text-xs font-bold text-slate-300 transition cursor-pointer"
              >
                Close Preview
              </button>
              <button
                type="button"
                onClick={() => window.print()}
                className="px-6 py-2.5 rounded-xl bg-[#FF4B32] hover:bg-[#e03a22] text-white text-xs font-black uppercase tracking-wider flex items-center gap-2 transition cursor-pointer shadow-lg shadow-[#FF4B32]/30"
              >
                <Printer className="w-4 h-4" />
                <span>Print / Save PDF Report</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}