--------------------------------------------------------------------------------
-- Quanty input file TEMPLATE for XMCD + NN dataset generation
-- Ni L2,3 XAS, Oh symmetry
--------------------------------------------------------------------------------

-- ---------- Parameters injected from Python (DO NOT EDIT) ----------
NElectrons_3d      = 8
TEN_DQ             = 0.8

SCALE_SLATER_3D    = 0.8
SCALE_SLATER_2P3D  = 0.8

SCALE_SOC_3D       = 1.3
SCALE_SOC_2P       = 1.1

H_EX               = 0.00333333333333

Prefix             = "Ni_L23_d8__10Dq0.8000__Sl0.8000__SOC2p1.1000__SOC3d1.3000__Hex0.003333"

-- ---------- General settings ----------
Verbosity(0x0000)
Temperature = 10
NPsis = 210
NPsisAuto = true
NConfigurations = 1

Emin = 845.0
Emax = 885.0
NPoints = 2000

ZeroShift = 5.7535
ExperimentalShift = 852.7

Gaussian = 0.0
Lorentzian = {{845.0, 0.48}, {885.0, 0.48}}
Gamma = 0.1

WaveVector = {0, 0, 1}
Ev = {0, 1, 0}
Eh = {1, 0, 0}

SpectraToCalculate = {"Absorption", "Circular Dichroic"}
DenseBorder = 2000
ShiftSpectra = true

-- ---------- Hamiltonian toggles ----------
AtomicTerm        = true
CrystalFieldTerm  = true
ExchangeFieldTerm = true

-- ---------- Basis ----------
NBosons = 0
NFermions = 16

NElectrons_2p = 6
IndexDn_2p = {0,2,4}
IndexUp_2p = {1,3,5}

IndexDn_3d = {6,8,10,12,14}
IndexUp_3d = {7,9,11,13,15}

-- ---------- Initialize Hamiltonians ----------
H_i = 0
H_f = 0

-- ---------- Number operators ----------
N_2p = NewOperator("Number", NFermions, IndexUp_2p, IndexUp_2p, {1,1,1})
     + NewOperator("Number", NFermions, IndexDn_2p, IndexDn_2p, {1,1,1})

N_3d = NewOperator("Number", NFermions, IndexUp_3d, IndexUp_3d, {1,1,1,1,1})
     + NewOperator("Number", NFermions, IndexDn_3d, IndexDn_3d, {1,1,1,1,1})

-- ---------- Atomic interactions ----------
F0_3d_3d = NewOperator("U", NFermions, IndexUp_3d, IndexDn_3d, {1,0,0})
F2_3d_3d = NewOperator("U", NFermions, IndexUp_3d, IndexDn_3d, {0,1,0})
F4_3d_3d = NewOperator("U", NFermions, IndexUp_3d, IndexDn_3d, {0,0,1})

F0_2p_3d = NewOperator("U", NFermions, IndexUp_2p, IndexDn_2p,
                       IndexUp_3d, IndexDn_3d, {1,0},{0,0})
F2_2p_3d = NewOperator("U", NFermions, IndexUp_2p, IndexDn_2p,
                       IndexUp_3d, IndexDn_3d, {0,1},{0,0})
G1_2p_3d = NewOperator("U", NFermions, IndexUp_2p, IndexDn_2p,
                       IndexUp_3d, IndexDn_3d, {0,0},{1,0})
G3_2p_3d = NewOperator("U", NFermions, IndexUp_2p, IndexDn_2p,
                       IndexUp_3d, IndexDn_3d, {0,0},{0,1})

-- Hartree–Fock values for Ni (from Crispy)
F2_3d_3d_i = 12.2338 * SCALE_SLATER_3D
F4_3d_3d_i = 7.5975  * SCALE_SLATER_3D
F0_3d_3d_i = 2/63 * (F2_3d_3d_i + F4_3d_3d_i)

F2_3d_3d_f = 13.0058 * SCALE_SLATER_3D
F4_3d_3d_f = 8.0839  * SCALE_SLATER_3D
F0_3d_3d_f = 2/63 * (F2_3d_3d_f + F4_3d_3d_f)

F2_2p_3d_f = 7.7213 * SCALE_SLATER_2P3D
G1_2p_3d_f = 5.7865 * SCALE_SLATER_2P3D
G3_2p_3d_f = 3.2909 * SCALE_SLATER_2P3D
F0_2p_3d_f = 1/15 * G1_2p_3d_f + 3/70 * G3_2p_3d_f

H_i = H_i + Chop(
  F0_3d_3d_i * F0_3d_3d +
  F2_3d_3d_i * F2_3d_3d +
  F4_3d_3d_i * F4_3d_3d
)

H_f = H_f + Chop(
  F0_3d_3d_f * F0_3d_3d +
  F2_3d_3d_f * F2_3d_3d +
  F4_3d_3d_f * F4_3d_3d +
  F0_2p_3d_f * F0_2p_3d +
  F2_2p_3d_f * F2_2p_3d +
  G1_2p_3d_f * G1_2p_3d +
  G3_2p_3d_f * G3_2p_3d
)

-- ---------- Spin–orbit coupling ----------
ldots_3d = NewOperator("ldots", NFermions, IndexUp_3d, IndexDn_3d)
ldots_2p = NewOperator("ldots", NFermions, IndexUp_2p, IndexDn_2p)

zeta_3d_i = 0.0826 * SCALE_SOC_3D
zeta_3d_f = 0.1022 * SCALE_SOC_3D
zeta_2p_f = 11.507 * SCALE_SOC_2P

H_i = H_i + Chop(zeta_3d_i * ldots_3d)
H_f = H_f + Chop(zeta_3d_f * ldots_3d + zeta_2p_f * ldots_2p)

-- ---------- Crystal field ----------
Akm = {{4,0,2.1},{4,-4,1.5*math.sqrt(0.7)},{4,4,1.5*math.sqrt(0.7)}}
tenDq_3d = NewOperator("CF", NFermions, IndexUp_3d, IndexDn_3d, Akm)

H_i = H_i + Chop(TEN_DQ * tenDq_3d)
H_f = H_f + Chop(TEN_DQ * tenDq_3d)

-- ---------- Angular momentum operators ----------
Sz   = NewOperator("Sz",   NFermions, IndexUp_3d, IndexDn_3d)
Lz   = NewOperator("Lz",   NFermions, IndexUp_3d, IndexDn_3d)
Ssqr = NewOperator("Ssqr", NFermions, IndexUp_3d, IndexDn_3d)
Lsqr = NewOperator("Lsqr", NFermions, IndexUp_3d, IndexDn_3d)
Jsqr = NewOperator("Jsqr", NFermions, IndexUp_3d, IndexDn_3d)

-- ---------- Exchange field ----------
H_i = H_i + Chop(H_EX * Sz)
H_f = H_f + Chop(H_EX * Sz)

-- ---------- Restrictions ----------
InitialRestrictions = {
  NFermions, NBosons,
  {"111111 0000000000", NElectrons_2p, NElectrons_2p},
  {"000000 1111111111", NElectrons_3d, NElectrons_3d}
}


--------------------------------------------------------------------------------
-- ================= Helper Functions =================
--------------------------------------------------------------------------------

function ValueInTable(Value, Table)
    for _, v in ipairs(Table) do
        if Value == v then return true end
    end
    return false
end

function GetSpectrum(G, Ids, dZ, NOperators, NPsis)
    if not (type(Ids) == "table") then Ids = {Ids} end
    local Id = 1
    local dZs = {}
    for i = 1, NOperators do
        for _ = 1, NPsis do
            if ValueInTable(i, Ids) then
                table.insert(dZs, dZ[Id])
            else
                table.insert(dZs, 0)
            end
            Id = Id + 1
        end
    end
    return Spectra.Sum(G, dZs)
end

function CalculateT(Basis, Eps)
    local T = 0
    for i = 1, #Basis do
        T = T + Eps[i] * Basis[i]
    end
    return Chop(T)
end

function WavefunctionsAndBoltzmannFactors(H, NPsis, NPsisAuto, Temperature, Threshold, StartRestrictions)
    if Threshold == nil then Threshold = 1e-8 end
    local Psis = Eigensystem(H, StartRestrictions, NPsis)
    if not (type(Psis) == "table") then Psis = {Psis} end
    local E0 = Psis[1] * H * Psis[1]
    local dZ = {}
    local Z = 0
    for i, Psi in ipairs(Psis) do
        local Ei = Psi * H * Psi
        if math.abs(Ei - E0) < Threshold^2 then
            dZ[i] = 1
        else
            dZ[i] = math.exp(-(Ei - E0) / Temperature)
        end
        Z = Z + dZ[i]
    end
    for i in ipairs(dZ) do dZ[i] = dZ[i] / Z end
    return Psis, dZ
end


function SaveSpectrum(G, Filename, Gaussian, Lorentzian, Pcl)
    if Pcl == nil then
        Pcl = 1
    end
    G = -1 / math.pi / Pcl * G
    G.Broaden(Gaussian, Lorentzian)
    G.Print({{"file", "../runs/" .. Filename .. ".spec"}})
end

--------------------------------------------------------------------------------
-- Solve initial Hamiltonian
--------------------------------------------------------------------------------
Temperature = Temperature * EnergyUnits.Kelvin.value
local Psis_i, dZ_i = WavefunctionsAndBoltzmannFactors(
    H_i, NPsis, NPsisAuto, Temperature, nil, InitialRestrictions
)

--------------------------------------------------------------------------------
-- NN labels (Boltzmann averaged)
--------------------------------------------------------------------------------
local Sz_avg, Lz_avg, Ssqr_avg, Lsqr_avg, Jsqr_avg = 0,0,0,0,0
for i, Psi in ipairs(Psis_i) do
    local w = dZ_i[i]
    Sz_avg   = Sz_avg   + w * Complex.Re(Psi * Sz   * Psi)
    Lz_avg   = Lz_avg   + w * Complex.Re(Psi * Lz   * Psi)
    Ssqr_avg = Ssqr_avg + w * Complex.Re(Psi * Ssqr * Psi)
    Lsqr_avg = Lsqr_avg + w * Complex.Re(Psi * Lsqr * Psi)
    Jsqr_avg = Jsqr_avg + w * Complex.Re(Psi * Jsqr * Psi)
end

--------------------------------------------------------------------------------
-- Save results to quanty_results.csv (with header)
--------------------------------------------------------------------------------
local result_file = "quanty_results_Ni_d8.csv"
local write_header = false
local fcheck = io.open(result_file, "r")
if fcheck == nil then
    write_header = true
else
    fcheck:close()
end

local meta = io.open(result_file, "a")
if meta then
    if write_header then
        meta:write(
            "Prefix,NElectrons_3d,TEN_DQ,SlaterScale,SOC_2p,SOC_3d,H_ex,"
            .. "Sz,Lz,S2,L2,J2\n"
        )
    end
    meta:write(string.format(
        "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        Prefix, NElectrons_3d, TEN_DQ,
        SCALE_SLATER_3D, SCALE_SOC_2P, SCALE_SOC_3D, H_EX,
        Sz_avg, Lz_avg, Ssqr_avg, Lsqr_avg, Jsqr_avg
    ))
    meta:close()
end

--------------------------------------------------------------------------------
-- ================= Spectra calculation =================
--------------------------------------------------------------------------------

local t = math.sqrt(1/2)

Tx_2p_3d = NewOperator("CF", NFermions, IndexUp_3d, IndexDn_3d,
                                 IndexUp_2p, IndexDn_2p, {{1,-1,t},{1,1,-t}})
Ty_2p_3d = NewOperator("CF", NFermions, IndexUp_3d, IndexDn_3d,
                                 IndexUp_2p, IndexDn_2p, {{1,-1,t*I},{1,1,t*I}})
Tz_2p_3d = NewOperator("CF", NFermions, IndexUp_3d, IndexDn_3d,
                                 IndexUp_2p, IndexDn_2p, {{1,0,1}})

Er = {t*(Eh[1]-I*Ev[1]), t*(Eh[2]-I*Ev[2]), t*(Eh[3]-I*Ev[3])}
El = {-t*(Eh[1]+I*Ev[1]), -t*(Eh[2]+I*Ev[2]), -t*(Eh[3]+I*Ev[3])}

Tops = {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}
Tr = CalculateT(Tops, Er)
Tl = CalculateT(Tops, El)
Tk = CalculateT(Tops, WaveVector)

SpectraAndOperators = {
    ["Absorption"] = {Tk},
    ["Circular Dichroic"] = {Tr, Tl},
}

local Tset = {}
for _, ops in pairs(SpectraAndOperators) do
    for _, op in ipairs(ops) do Tset[op] = true end
end

local Tlist = {}
for op, _ in pairs(Tset) do table.insert(Tlist, op) end

if ShiftSpectra then
    Emin = Emin - (ZeroShift + ExperimentalShift)
    Emax = Emax - (ZeroShift + ExperimentalShift)
end

G = CreateSpectra(H_f, Tlist, Psis_i,
    {{"Emin", Emin}, {"Emax", Emax}, {"NE", NPoints},
     {"Gamma", Gamma}, {"DenseBorder", DenseBorder}})

if ShiftSpectra then
    G.Shift(ZeroShift + ExperimentalShift)
end

local dZspec = {}
for _ in ipairs(Tlist) do
    for j in ipairs(Psis_i) do
        table.insert(dZspec, dZ_i[j])
    end
end

local Ids = {}
for k, v in pairs(Tlist) do Ids[v] = k end

for name, ops in pairs(SpectraAndOperators) do
    if ValueInTable(name, SpectraToCalculate) then
        local idx = {}
        for _, op in ipairs(ops) do table.insert(idx, Ids[op]) end

        if name == "Absorption" then
            Gk = GetSpectrum(G, idx, dZspec, #Tlist, #Psis_i)
            SaveSpectrum(Gk, Prefix .. "_k", Gaussian, Lorentzian)
        end

        if name == "Circular Dichroic" then
            Gr = GetSpectrum(G, idx[1], dZspec, #Tlist, #Psis_i)
            Gl = GetSpectrum(G, idx[2], dZspec, #Tlist, #Psis_i)
            SaveSpectrum(Gr - Gl, Prefix .. "_cd", Gaussian, Lorentzian)
        end
    end
end