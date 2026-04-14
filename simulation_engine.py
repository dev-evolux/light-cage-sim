import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import re

# === UTILIDADES MATEMÁTICAS ===

def wavelength_to_rgb(wavelength, gamma=0.8):
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = (-(wavelength - 440) / (440 - 380)) * attenuation
        G, B = 0.0, 1.0 * attenuation
    elif wavelength >= 440 and wavelength <= 490:
        R, G, B = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R, G, B = 0.0, 1.0, -(wavelength - 510) / (510 - 490)
    elif wavelength >= 510 and wavelength <= 580:
        R, G, B = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R, G, B = 1.0, -(wavelength - 645) / (645 - 580), 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R, G, B = 1.0 * attenuation, 0.0, 0.0
    else:
        R, G, B = 0.0, 0.0, 0.0
    return (R, G, B)

def fresnel_transmission(n1, n2, cos_theta_i, cos_theta_t):
    rs = ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t))**2
    rp = ((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i))**2
    return 1.0 - 0.5 * (rs + rp)

def normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norm + 1e-16)

def rotate_3d(vectors, rx_deg, ry_deg, rz_deg):
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)
    
    cx, sx = np.cos(rx), np.sin(rx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    
    cy, sy = np.cos(ry), np.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    return vectors @ R.T

# === CLASES DE PARSEO (TM33 e IES) ===

class TM33Parser:
    def __init__(self, xml_content):
        self.is_ies = False
        xml_clean = re.sub(r'\sxmlns="[^"]+"', '', xml_content, count=1)
        xml_clean = xml_clean.replace('xsi:', '')
        
        try:
            self.root = ET.fromstring(xml_clean)
            self.lum_interp = self._create_interpolator("./Emitter/LuminousData/LuminousIntensity")
            self.rad_interp = self._create_interpolator("./Emitter/RadiantData/RadiantIntensity")
        except Exception as e:
            print(f"Error XML parsing: {e}")
            self.root = ET.Element("root")
            self.lum_interp = lambda x: np.ones(len(x))
            self.rad_interp = lambda x: np.ones(len(x))

    def _create_interpolator(self, xpath):
        node = self.root.find(xpath)
        if node is None:
            tag = xpath.split('/')[-1]
            for elem in self.root.iter():
                if elem.tag.endswith(tag):
                    node = elem
                    break
        
        if node is None: return lambda x: np.zeros(len(x))

        data, h_set, v_set = [], set(), set()
        for d in node.findall('.//IntData'):
             h = float(d.get('h', 0))
             v = float(d.get('v', 0))
             try: val = float(d.text)
             except: val = 0.0
             data.append((h, v, val))
             h_set.add(h); v_set.add(v)

        if not data: return lambda x: np.zeros(len(x))

        sorted_h, sorted_v = sorted(list(h_set)), sorted(list(v_set))
        grid = np.zeros((len(sorted_h), len(sorted_v)))
        
        h_idx = {val: i for i, val in enumerate(sorted_h)}
        v_idx = {val: i for i, val in enumerate(sorted_v)}

        for h, v, val in data:
            grid[h_idx[h], v_idx[v]] = val

        if 0.0 in h_idx and 360.0 not in h_idx:
            sorted_h.append(360.0)
            grid = np.vstack([grid, grid[0:1, :]])

        return RegularGridInterpolator((sorted_h, sorted_v), grid, bounds_error=False, fill_value=0)

    def get_spectrum(self):
        spectrum = {}
        for tag in ["EmitterSpectral", "SpectralData", "Spectral"]:
            node = self.root.find(f".//{tag}")
            if node is not None:
                for pwr in node.findall(".//PwrData"):
                    try:
                        w = float(pwr.get('w'))
                        val = float(pwr.text)
                        spectrum[w] = val
                    except: pass
                if spectrum: return spectrum
        return spectrum

    def get_intensity(self, vectors):
        vz = np.clip(-vectors[:, 2], -1.0, 1.0) 
        theta_rad = np.arccos(vz)
        v_deg = np.degrees(theta_rad)
        
        phi_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        h_deg = np.mod(np.degrees(phi_rad), 360)
        
        pts = np.column_stack((h_deg, v_deg))
        return self.lum_interp(pts), self.rad_interp(pts)

class IESParser:
    def __init__(self, content_str):
        self.is_ies = True
        lines = content_str.replace('\r', '\n').split('\n')
        data_lines = []
        in_data = False
        tilt_type = "NONE"
        
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith('TILT='):
                tilt_type = line.split('=')[1].strip()
                in_data = True
                continue
            if in_data:
                data_lines.append(line)
        
        # Fallback si no hay header clásico TILT
        if not in_data:
            for i, line in enumerate(lines):
                if re.match(r'^\s*[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s*', line):
                    data_lines = lines[i:]
                    break
        
        tokens = []
        for line in data_lines:
            tokens.extend(line.split())
        
        if not tokens:
            self.lum_interp = lambda x: np.zeros(len(x))
            self.rad_interp = lambda x: np.zeros(len(x))
            return
            
        idx = 0
        if tilt_type == "INCLUDE":
            num_tilt_angles = int(tokens[1])
            idx = 2 + 2 * num_tilt_angles 
            
        self.num_lamps = int(tokens[idx])
        self.lumens = float(tokens[idx+1])
        self.multiplier = float(tokens[idx+2])
        num_v = int(tokens[idx+3])
        num_h = int(tokens[idx+4])
        
        idx += 13 # Salto a los ángulos verticales (estándar LM-63)
        v_angles = [float(x) for x in tokens[idx:idx+num_v]]
        idx += num_v
        h_angles = [float(x) for x in tokens[idx:idx+num_h]]
        idx += num_h
        
        candelas = np.zeros((num_h, num_v))
        for i in range(num_h):
            for j in range(num_v):
                candelas[i, j] = float(tokens[idx]) * self.multiplier
                idx += 1
                
        h_angles = np.array(h_angles)
        v_angles = np.array(v_angles)
        
        # Algoritmo de expansión de Simetrías IES (360 grados, Cuadrante, Bilateral)
        if len(h_angles) == 1 or h_angles[-1] == 0:
            h_angles = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
            candelas = np.tile(candelas[0, :], (5, 1))
        elif h_angles[-1] == 90:
            h_angles_180 = np.concatenate((h_angles, 180 - h_angles[-2::-1]))
            candelas_180 = np.vstack((candelas, candelas[-2::-1, :]))
            h_angles_full = np.concatenate((h_angles_180, 360 - h_angles_180[-2::-1]))
            candelas_full = np.vstack((candelas_180, candelas_180[-2::-1, :]))
            h_angles, candelas = h_angles_full, candelas_full
        elif h_angles[-1] == 180:
            h_angles_full = np.concatenate((h_angles, 360 - h_angles[-2::-1]))
            candelas_full = np.vstack((candelas, candelas[-2::-1, :]))
            h_angles, candelas = h_angles_full, candelas_full
        elif h_angles[-1] < 360:
            h_angles = np.append(h_angles, 360.0)
            candelas = np.vstack((candelas, candelas[0, :]))
            
        self.lum_interp = RegularGridInterpolator((h_angles, v_angles), candelas, bounds_error=False, fill_value=0)
        # Asignamos la misma interpolación. En el 'run', si is_ies=True, esto será normalizado a Radiometría automáticamente.
        self.rad_interp = self.lum_interp

    def get_spectrum(self):
        return {} # Los archivos IES nativos no tienen espectro.

    def get_intensity(self, vectors):
        vz = np.clip(-vectors[:, 2], -1.0, 1.0) 
        theta_rad = np.arccos(vz)
        v_deg = np.degrees(theta_rad)
        
        phi_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        h_deg = np.mod(np.degrees(phi_rad), 360)
        
        pts = np.column_stack((h_deg, v_deg))
        return self.lum_interp(pts), self.rad_interp(pts)

# === MOTOR FÍSICO ===

class SimulationEngine:
    def __init__(self):
        self.parsers = {} 
    
    def load_file(self, filename, content_str):
        try:
            if filename.lower().endswith('.ies'):
                parser = IESParser(content_str)
            else:
                parser = TM33Parser(content_str)
            self.parsers[filename] = parser
            return True
        except Exception as e:
            print(f"Error cargando {filename}: {e}")
            return False

    def run(self, config):
        env = config['env']
        env_type = env.get('type', 'estanque')
        env_x = float(env['x'])
        env_y = float(env['y'])
        
        n1 = float(env.get('n1', 1.0))
        n2 = float(env.get('n2', 1.333))

        z_interface = 0.0 if env_type == 'jaula' else float(env.get('z_interface', 3.2))

        kd_measurable = config.get('kd', {})
        target_depths_input = config.get('target_depths', [2.0])
        n_rays = int(config.get('rays', 50000))
        
        avg_kd = 0.2
        if 'fijo' in kd_measurable:
            avg_kd = float(kd_measurable['fijo'])

        results = {str(d): {'x': [], 'y': [], 'val': [], 'lamp_idx': []} for d in target_depths_input}

        for i_lamp, lamp in enumerate(config['lamps']):
            xml_id = lamp['xml']
            if xml_id not in self.parsers: continue
            
            parser = self.parsers[xml_id]
            
            pos_z = -float(lamp['z']) if env_type == 'jaula' else float(lamp['z'])
            pos = np.array([float(lamp['x']), float(lamp['y']), pos_z])
            dimming = float(lamp['dim'])
            
            rot_x = float(lamp.get('rot_x', 0))
            rot_y = float(lamp.get('rot_y', 0))
            rot_z = float(lamp.get('rot_z', 0))

            indices = np.arange(0, n_rays, dtype=float) + 0.5
            phi = np.arccos(1 - 2*indices/n_rays) 
            theta = np.pi * (1 + 5**0.5) * indices 

            lx = np.sin(phi) * np.cos(theta)
            ly = np.sin(phi) * np.sin(theta)
            lz = -np.abs(np.cos(phi))

            rays_local = np.column_stack((lx, ly, lz))

            lum, rad = parser.get_intensity(rays_local)
            
            # NORMALIZACIÓN RADIOMÉTRICA IES
            # Extraemos la distribución polar y forzamos su energía a calzar con el Input de Potencia de la UI.
            if getattr(parser, 'is_ies', False):
                total_current_power = np.sum(rad) * (4 * np.pi / n_rays)
                user_power = float(lamp.get('power', 600))
                if total_current_power > 0:
                    rad = rad * (user_power / total_current_power)

            mask = rad > 0
            rays_local = rays_local[mask]
            flux_rad = rad[mask] * (4 * np.pi / n_rays) * dimming

            rays_global = rays_local
            if rot_x != 0 or rot_y != 0 or rot_z != 0:
                rays_global = rotate_3d(rays_local, rot_x, rot_y, rot_z)

            for orig_depth in target_depths_input:
                depth = -float(orig_depth) if env_type == 'jaula' else float(orig_depth)
                
                down_mask = rays_global[:, 2] < -1e-6
                valid_rays = rays_global[down_mask]
                valid_flux = flux_rad[down_mask]

                if len(valid_rays) == 0: continue

                if pos[2] > z_interface and depth <= z_interface:
                    t_int = (z_interface - pos[2]) / valid_rays[:, 2]
                    P_int = pos + valid_rays * t_int[:, np.newaxis]

                    cos_theta_i = -valid_rays[:, 2] 
                    sin2_theta_t = (n1/n2)**2 * (1.0 - cos_theta_i**2)

                    tir_mask = sin2_theta_t <= 1.0
                    if not np.any(tir_mask): continue

                    v_rays = valid_rays[tir_mask]
                    v_flux = valid_flux[tir_mask]
                    P_int = P_int[tir_mask]
                    c_ti = cos_theta_i[tir_mask]
                    s2_tt = sin2_theta_t[tir_mask]

                    c_tt = np.sqrt(1.0 - s2_tt)
                    T_vec = (n1/n2) * v_rays + ((n1/n2) * c_ti - c_tt)[:, np.newaxis] * np.array([0, 0, 1])
                    T_fresnel = fresnel_transmission(n1, n2, c_ti, c_tt)

                    t_water = (depth - z_interface) / T_vec[:, 2]
                    impact_x = P_int[:, 0] + T_vec[:, 0] * t_water
                    impact_y = P_int[:, 1] + T_vec[:, 1] * t_water
                    
                    d_w = np.linalg.norm(T_vec * t_water[:, np.newaxis], axis=1)
                    transmission = np.exp(-avg_kd * d_w)
                    val = v_flux * T_fresnel * transmission * 0.98

                elif pos[2] <= z_interface and depth <= z_interface:
                    t = (depth - pos[2]) / valid_rays[:, 2]
                    impact_x = pos[0] + valid_rays[:, 0] * t
                    impact_y = pos[1] + valid_rays[:, 1] * t
                    
                    d_w = np.linalg.norm(valid_rays * t[:, np.newaxis], axis=1)
                    transmission = np.exp(-avg_kd * d_w)
                    val = valid_flux * transmission * 0.98
                    
                else:
                    t = (depth - pos[2]) / valid_rays[:, 2]
                    impact_x = pos[0] + valid_rays[:, 0] * t
                    impact_y = pos[1] + valid_rays[:, 1] * t
                    val = valid_flux * 0.98

                in_bounds = (impact_x >= 0) & (impact_x <= env_x) & (impact_y >= 0) & (impact_y <= env_y)
                valid_indices = np.where(in_bounds)[0]

                results[str(orig_depth)]['x'].extend(impact_x[valid_indices].tolist())
                results[str(orig_depth)]['y'].extend(impact_y[valid_indices].tolist())
                results[str(orig_depth)]['val'].extend(val[valid_indices].tolist())
                results[str(orig_depth)]['lamp_idx'].extend(np.full(len(valid_indices), i_lamp).tolist())

        return results