from flask import Flask, render_template, jsonify, request
import os
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker 

try:
    trapz_func = np.trapezoid
except AttributeError:
    trapz_func = np.trapz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from simulation_engine import SimulationEngine

app = Flask(__name__)
engine = SimulationEngine()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm", 
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150 
})

UPLOAD_FOLDER = './uploaded_lamps'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('simulation.html')

@app.route('/api/upload_lamp', methods=['POST'])
def upload_lamp():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    if file:
        filename = file.filename
        content = file.read().decode('utf-8', errors='ignore')
        success = engine.load_file(filename, content)
        if success:
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
            return jsonify({"status": "ok", "filename": filename, "msg": "Lámpara cargada exitosamente"})
        else:
            return jsonify({"status": "error", "msg": "El archivo no es válido"}), 500

@app.route('/api/get_lamps', methods=['GET'])
def get_lamps():
    if not os.path.exists(UPLOAD_FOLDER): return jsonify({"status": "ok", "lamps": []})
    lamps = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.xml') or f.lower().endswith('.ies')]
    return jsonify({"status": "ok", "lamps": lamps})

@app.route('/api/lamp_profile/<filename>')
def lamp_profile(filename):
    try:
        parser = engine.parsers.get(filename)
        if not parser:
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    engine.load_file(filename, f.read())
                parser = engine.parsers.get(filename)
        
        if not parser: return jsonify({"error": "Lámpara no encontrada en memoria"})
        
        theta_arr = np.linspace(0, 180, 73)
        v_rad = np.radians(theta_arr)
        
        c0 = np.column_stack((np.sin(v_rad), np.zeros_like(v_rad), -np.cos(v_rad)))
        c180 = np.column_stack((-np.sin(v_rad), np.zeros_like(v_rad), -np.cos(v_rad)))
        c90 = np.column_stack((np.zeros_like(v_rad), np.sin(v_rad), -np.cos(v_rad)))
        c270 = np.column_stack((np.zeros_like(v_rad), -np.sin(v_rad), -np.cos(v_rad)))
        
        _, rad0 = parser.get_intensity(c0)
        _, rad180 = parser.get_intensity(c180)
        _, rad90 = parser.get_intensity(c90)
        _, rad270 = parser.get_intensity(c270)
        
        max_rad = max(np.max(rad0), np.max(rad180), np.max(rad90), np.max(rad270))
        if max_rad == 0: max_rad = 1.0
        
        plane_0_180_theta = np.concatenate((theta_arr, -theta_arr[::-1]))
        plane_0_180_rad = np.concatenate((rad0, rad180[::-1])) / max_rad
        plane_90_270_theta = np.concatenate((theta_arr, -theta_arr[::-1]))
        plane_90_270_rad = np.concatenate((rad90, rad270[::-1])) / max_rad
        
        return jsonify({
            "c0_180": {"theta": plane_0_180_theta.tolist(), "rad": plane_0_180_rad.tolist()},
            "c90_270": {"theta": plane_90_270_theta.tolist(), "rad": plane_90_270_rad.tolist()}
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/calc_kd', methods=['POST'])
def calc_kd():
    try:
        data = request.json
        target_x, target_y = float(data['x']), float(data['y'])
        measurements = data['measurements']
        
        pts = [m for m in measurements if abs(m['x'] - target_x) < 0.1 and abs(m['y'] - target_y) < 0.1]
        pts.sort(key=lambda p: float(p['z']))
        
        results = []
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                z1, val1 = float(pts[i]['z']), float(pts[i]['val'])
                z2, val2 = float(pts[j]['z']), float(pts[j]['val'])
                if val1 > 0 and val2 > 0 and abs(z2 - z1) > 0.001:
                    kd = (np.log(val1) - np.log(val2)) / abs(z2 - z1)
                    results.append({"z1": z1, "val1": val1, "z2": z2, "val2": val2, "kd": kd})
        return jsonify({"status": "ok", "kds": results})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

def _plot_map_on_ax(ax, E, X, Y, config, env_type, center_x, center_y, env_radio, env_x, env_y, contour_val, max_irr_local, roi, depth_val):
    scale_type = config.get('color_scale_type', 'log')
    
    if scale_type == 'log':
        vmin = contour_val if contour_val > 0 else 1e-4
        vmax = max_irr_local if max_irr_local > vmin else vmin + 1.0
        
        E_plot = np.maximum(E, vmin)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 25)
    else:
        vmin = 0.0
        vmax = max_irr_local if max_irr_local > vmin else vmin + 1.0
        E_plot = E
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, 25)
        
    cmap = plt.cm.YlGnBu_r.copy()
    if scale_type == 'log':
        cmap.set_under('#ffffff')
    
    cf = ax.contourf(X, Y, E_plot, levels=levels, cmap=cmap, norm=norm, extend='min' if scale_type == 'log' else 'neither')
    
    if config.get('draw_contour') and np.max(E) >= contour_val:
        try:
            CS_high = ax.contour(X, Y, E, levels=[contour_val], colors='lime', linewidths=2.5)
            ax.clabel(CS_high, inline=True, fontsize=9, fmt=f'{contour_val}', colors='lime')
        except Exception: pass

    env_shape = config['env'].get('shape', 'circle' if env_type == 'estanque' else 'rect')

    if env_shape == 'circle':
        roi_circle = plt.Circle((center_x, center_y), env_radio, edgecolor='cyan', facecolor='none', linestyle='--', linewidth=2)
        ax.add_patch(roi_circle)
        ax.plot(center_x, center_y, '+', color='cyan', markersize=10)
    else:
        rect = plt.Rectangle((0, 0), env_x, env_y, edgecolor='cyan', facecolor='none', linestyle='--', linewidth=2)
        ax.add_patch(rect)

    if roi.get('type') == 'paralelepipedo':
        if abs(depth_val - float(roi.get('cz', 0))) <= float(roi.get('h', 0)) / 2.0:
            rx = float(roi['cx']) - float(roi['l']) / 2
            ry = float(roi['cy']) - float(roi['w']) / 2
            r_rect = plt.Rectangle((rx, ry), float(roi['l']), float(roi['w']), edgecolor='magenta', facecolor='none', linestyle='-.', linewidth=2.5)
            ax.add_patch(r_rect)
    elif roi.get('type') == 'cilindro':
        if abs(depth_val - float(roi.get('cz', 0))) <= float(roi.get('h', 0)) / 2.0:
            circ = plt.Circle((float(roi['cx']), float(roi['cy'])), float(roi['r']), edgecolor='magenta', facecolor='none', linestyle='-.', linewidth=2.5)
            ax.add_patch(circ)

    for lamp in config['lamps']:
        ax.plot(float(lamp['x']), float(lamp['y']), marker='*', color='yellow', markeredgecolor='black', markersize=10)

    ax.set_aspect('equal')
    ax.set_xlim(0, env_x)
    ax.set_ylim(0, env_y)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return cf

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    try:
        config = request.json
        env_x, env_y = float(config['env']['x']), float(config['env']['y'])
        env_z = float(config['env'].get('z', 15.0))
        env_type = config['env'].get('type', 'estanque')
        env_shape = config['env'].get('shape', 'circle' if env_type == 'estanque' else 'rect')
        raw_radio = config['env'].get('radio')
        env_radio = float(raw_radio) if raw_radio is not None else env_x / 2.0
        roi = config.get('roi', {'type': 'global'})

        contour_val = config.get('contour_val', 0.017)
        target_depths_requested = sorted([float(d) for d in config.get('target_depths', [])], reverse=True)
        kds_requested = config.get('kd_list', [0.2])
        aporte_puntos = config.get('aporte_puntos', [])
        
        # Limitar la profundidad de evaluación según el modo de simulación
        if roi['type'] in ['paralelepipedo', 'cilindro']:
            calc_min_z = max(0, float(roi.get('cz', 0)) - float(roi.get('h', 0)) / 2.0)
            calc_max_z = float(roi.get('cz', 0)) + float(roi.get('h', 0)) / 2.0
        else:
            calc_min_z = 0.0
            # Si es estanque, la profundidad global está dada por la altura del agua
            calc_max_z = float(config['env'].get('z_interface', 3.2)) if env_type == 'estanque' else env_z
            
        profile_step = float(config.get('profile_step', 0.5))
        prof_d = np.arange(calc_min_z, calc_max_z + profile_step, profile_step)
        
        all_depths_set = set(target_depths_requested)
        if config.get('plot_depth_profile'):
            all_depths_set.update(prof_d.tolist())
            
        all_depths_requested = sorted(list(all_depths_set), reverse=True)
        config['target_depths'] = all_depths_requested
        
        results_by_kd = {}
        table_data = []
        spectrum_results = {}
        lamps_names = [lamp['xml'] for lamp in config['lamps']]
        
        if config.get('plot_spectrum'):
            ranges = config.get('spectrum_ranges', {'blue': [400, 499], 'green': [500, 599], 'red': [600, 750]})
            for xml_name in config.get('spectrum_lamps', []):
                parser = engine.parsers.get(xml_name)
                if parser:
                    spectrum = parser.get_spectrum()
                    if spectrum:
                        wls, pwrs = np.array(sorted(spectrum.keys())), np.array([spectrum[w] for w in sorted(spectrum.keys())])
                        total_auc = trapz_func(pwrs, wls)
                        if total_auc == 0: total_auc = 1e-9
                        
                        fig_spec, ax_spec = plt.subplots(figsize=(7, 4))
                        points = np.array([wls, pwrs]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        norm = plt.Normalize(380, 780)
                        lc = LineCollection(segments, cmap='turbo', norm=norm)
                        lc.set_array(wls)
                        lc.set_linewidth(2.5)
                        ax_spec.add_collection(lc)
                        
                        colors, labels_es = {'blue': '#1f77b4', 'green': '#2ca02c', 'red': '#d62728'}, {'blue': 'Azul', 'green': 'Verde', 'red': 'Rojo'}
                        for color_name, (w_min, w_max) in ranges.items():
                            mask = (wls >= w_min) & (wls <= w_max)
                            if np.any(mask):
                                pct = (trapz_func(pwrs[mask], wls[mask]) / total_auc) * 100
                                ax_spec.fill_between(wls, pwrs, where=mask, color=colors.get(color_name, 'gray'), alpha=0.3, label=rf"{labels_es.get(color_name, color_name)} ({w_min}-{w_max}nm): {pct:.1f}%")
                        
                        ax_spec.set_title(rf"Espectro - {xml_name}")
                        ax_spec.set_xlabel(r"Longitud de onda $[nm]$")
                        ax_spec.set_ylabel(r"Potencia Relativa")
                        ax_spec.legend(loc='upper right', fontsize=9)
                        ax_spec.grid(True, linestyle=':', alpha=0.6)
                        ax_spec.set_xlim(380, 780)
                        ax_spec.set_ylim(0, np.max(pwrs) * 1.1)
                        
                        buf_spec = io.BytesIO()
                        plt.savefig(buf_spec, format='png', bbox_inches='tight', transparent=True)
                        plt.close(fig_spec)
                        spectrum_results[xml_name] = base64.b64encode(buf_spec.getvalue()).decode('utf-8')

        bins = 100
        grid_x = np.linspace(0, env_x, bins)
        grid_y = np.linspace(0, env_y, bins)
        X, Y = np.meshgrid((grid_x[:-1]+grid_x[1:])/2, (grid_y[:-1]+grid_y[1:])/2)
        x_centers, y_centers = (grid_x[:-1] + grid_x[1:]) / 2, (grid_y[:-1] + grid_y[1:]) / 2
        area_bin = (grid_x[1]-grid_x[0]) * (grid_y[1]-grid_y[0])
        center_x, center_y = env_x / 2.0, env_y / 2.0
        
        for kd_val in kds_requested:
            kd_val = float(kd_val)
            config['kd'] = {'fijo': kd_val}
            
            raw_results = engine.run(config)
            
            kd_res = {"depths": {}, "combined_image": "", "comparison_image": "", "depth_profile_image": "", "aportes": []}
            
            layer_stats = []
            max_irr_all, min_irr_all = -1, 999999
            
            fig_comb, axes_comb = plt.subplots(1, len(target_depths_requested), figsize=(7 * len(target_depths_requested), 6), constrained_layout=True)
            if len(target_depths_requested) == 1: axes_comb = [axes_comb]
            target_idx = 0

            comp_z, comp_meas, comp_sim = [], [], []

            for depth_val in all_depths_requested:
                depth_str = str(depth_val)
                data = None
                for k in raw_results.keys():
                    if abs(float(k) - depth_val) < 0.01:
                        data = raw_results[k]
                        break
                
                is_target = any(abs(depth_val - td) < 1e-4 for td in target_depths_requested)

                if data is None or not data['x']:
                    if is_target:
                        kd_res["depths"][depth_str] = {"image": "", "max": 0, "avg": 0, "area_ilum": 0}
                        axes_comb[target_idx].set_title(f"Z = {depth_val}m (Sin datos)")
                        axes_comb[target_idx].axis('off')
                        target_idx += 1
                    continue
                    
                pts = np.column_stack((data['x'], data['y']))
                vals = np.array(data['val'])
                lamp_idxs = np.array(data.get('lamp_idx', []))

                H, _, _ = np.histogram2d(pts[:,0], pts[:,1], bins=[grid_x, grid_y], weights=vals)
                E = H.T / area_bin

                mask = np.ones_like(E, dtype=bool)
                z_valid = True
                
                if roi['type'] == 'paralelepipedo':
                    cx, cy, cz = float(roi['cx']), float(roi['cy']), float(roi['cz'])
                    l, w, h = float(roi['l']), float(roi['w']), float(roi['h'])
                    if abs(depth_val - cz) <= h / 2.0:
                        mask = (np.abs(X - cx) <= l / 2.0) & (np.abs(Y - cy) <= w / 2.0)
                    else:
                        z_valid = False
                        mask = np.zeros_like(E, dtype=bool)
                elif roi['type'] == 'cilindro':
                    cx, cy, cz = float(roi['cx']), float(roi['cy']), float(roi['cz'])
                    r_roi, h = float(roi['r']), float(roi['h'])
                    if abs(depth_val - cz) <= h / 2.0:
                        mask = ((X - cx)**2 + (Y - cy)**2) <= r_roi**2
                    else:
                        z_valid = False
                        mask = np.zeros_like(E, dtype=bool)
                else: 
                    # Ignorar env_z si es estanque (ya que la altura está definida por z_interface)
                    if env_type != 'estanque' and depth_val > env_z: 
                        z_valid = False
                        mask = np.zeros_like(E, dtype=bool)
                    elif env_type == 'estanque' and depth_val > float(config['env'].get('z_interface', 3.2)):
                        z_valid = False
                        mask = np.zeros_like(E, dtype=bool)
                    elif env_shape == 'circle':
                        mask = ((X - center_x)**2 + (Y - center_y)**2) <= env_radio**2
                    else:
                        mask = np.ones_like(E, dtype=bool)
                
                area_total_layer = np.sum(mask) * area_bin
                
                if z_valid and np.any(mask):
                    E_roi = E[mask]
                    avg_irr, min_irr, max_irr = np.mean(E_roi), np.min(E_roi), np.max(E_roi)
                    area_ilum = np.sum(E_roi >= contour_val) * area_bin
                    
                    max_irr_all = max(max_irr_all, max_irr)
                    min_irr_all = min(min_irr_all, min_irr)
                else:
                    avg_irr, min_irr, max_irr, area_ilum = 0, 0, 0, 0
                
                if z_valid:
                    layer_stats.append({
                        'z': depth_val, 'avg': avg_irr, 'area': area_ilum, 'tot': area_total_layer
                    })

                label_area = "Vol. ROI" if roi['type'] != 'global' else ("Estanque" if env_type == 'estanque' else "Area Total")

                if is_target:
                    pts_at_depth = [p for p in aporte_puntos if abs(float(p['z']) - depth_val) < 0.1]
                    if pts_at_depth:
                        interp_tot = RegularGridInterpolator((x_centers, y_centers), H / area_bin, bounds_error=False, fill_value=0)
                        E_lamps_interp = []
                        
                        for i_lamp in range(len(config['lamps'])):
                            mask_i = (lamp_idxs == i_lamp)
                            if np.any(mask_i):
                                H_i, _, _ = np.histogram2d(pts[mask_i,0], pts[mask_i,1], bins=[grid_x, grid_y], weights=vals[mask_i])
                                E_lamps_interp.append(RegularGridInterpolator((x_centers, y_centers), H_i / area_bin, bounds_error=False, fill_value=0))
                            else:
                                E_lamps_interp.append(None)
                                
                        for p in pts_at_depth:
                            tot_val = float(interp_tot((p['x'], p['y'])))
                            lamp_vals = []
                            for i_lamp, interp_i in enumerate(E_lamps_interp):
                                val_i = float(interp_i((p['x'], p['y']))) if interp_i is not None else 0.0
                                pct = (val_i / tot_val * 100) if tot_val > 0 else 0
                                lamp_vals.append({'lamp_idx': i_lamp, 'val': val_i, 'pct': pct})
                            
                            kd_res["aportes"].append({
                                'x': p['x'], 'y': p['y'], 'z': p['z'], 'total': tot_val, 'lamps': lamp_vals
                            })

                if is_target and config.get('compare_measurements') and config.get('compare_x'):
                    m_pts = [m for m in config.get('measurements', []) if abs(m['x'] - config['compare_x']) < 0.1 and abs(m['y'] - config['compare_y']) < 0.1 and abs(float(m['z']) - depth_val) < 0.1]
                    if m_pts:
                        interp = RegularGridInterpolator((x_centers, y_centers), H / area_bin, bounds_error=False, fill_value=0)
                        sim_val = interp((config['compare_x'], config['compare_y']))
                        avg_meas_val = np.mean([float(m['val']) for m in m_pts])
                        comp_z.append(depth_val); comp_meas.append(avg_meas_val); comp_sim.append(float(sim_val))

                if is_target:
                    fig_ind, ax_ind = plt.subplots(figsize=(7, 6), constrained_layout=True)
                    cf_ind = _plot_map_on_ax(ax_ind, E, X, Y, config, env_type, center_x, center_y, env_radio, env_x, env_y, contour_val, max_irr, roi, depth_val)
                    plt.colorbar(cf_ind, ax=ax_ind, label="$W/m^2$", shrink=0.6, aspect=35, format="%.3f")
                    
                    stats_text = f"Stats {label_area}:\nProm: {avg_irr:.4f} W/m²\nMin: {min_irr:.4f}\nMax: {max_irr:.4f}\nÁrea >= {contour_val}: {area_ilum:.1f} m²"
                    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                    ax_ind.text(0.02, 0.98, stats_text, transform=ax_ind.transAxes, fontsize=10, verticalalignment='top', bbox=props)
                    
                    buf_ind = io.BytesIO()
                    plt.savefig(buf_ind, format='png', bbox_inches='tight', transparent=False)
                    plt.close(fig_ind)
                    kd_res["depths"][depth_str] = {"image": base64.b64encode(buf_ind.getvalue()).decode('utf-8')}

                    ax_comb = axes_comb[target_idx]
                    cf_comb = _plot_map_on_ax(ax_comb, E, X, Y, config, env_type, center_x, center_y, env_radio, env_x, env_y, contour_val, max_irr, roi, depth_val)
                    ax_comb.set_title(f"Z = {depth_val}m", fontsize=12)
                    plt.colorbar(cf_comb, ax=ax_comb, shrink=0.5, aspect=20, format="%.3f")
                    target_idx += 1

            valid_stats = [s for s in layer_stats if calc_min_z - 1e-3 <= s['z'] <= calc_max_z + 1e-3]
            valid_stats.sort(key=lambda x: x['z']) 
            
            if len(valid_stats) > 1:
                z_arr = np.array([s['z'] for s in valid_stats])
                area_ilum_arr = np.array([s['area'] for s in valid_stats])
                area_tot_arr = np.array([s['tot'] for s in valid_stats])
                avg_irr_arr = np.array([s['avg'] for s in valid_stats])
                
                vol_ilum_total = trapz_func(area_ilum_arr, z_arr)
                vol_tot_total = trapz_func(area_tot_arr, z_arr)
                
                vol_pct = (vol_ilum_total / vol_tot_total) * 100 if vol_tot_total > 0 else 0
                avg_all = trapz_func(avg_irr_arr * area_tot_arr, z_arr) / vol_tot_total if vol_tot_total > 0 else 0
            else:
                vol_pct = (valid_stats[0]['area'] / valid_stats[0]['tot']) * 100 if len(valid_stats) > 0 and valid_stats[0]['tot'] > 0 else 0
                avg_all = valid_stats[0]['avg'] if len(valid_stats) > 0 else 0

            depths_txt = " y ".join([str(d) for d in target_depths_requested])
            fig_comb.suptitle(f"Irradiancia simulada a {depths_txt} m del fondo (Kd={kd_val})", fontsize=16, fontfamily='serif')
            buf_comb = io.BytesIO()
            plt.savefig(buf_comb, format='png', bbox_inches='tight', transparent=False)
            plt.close(fig_comb)
            kd_res["combined_image"] = base64.b64encode(buf_comb.getvalue()).decode('utf-8')

            if config.get('plot_depth_profile') and len(valid_stats) > 0:
                z_vals = [s['z'] for s in valid_stats]
                cum_irr_vals = []
                cum_vol_pct = []
                v_ilum_run = 0.0
                v_tot_run = 0.0
                EA_run = 0.0
                
                for i in range(len(valid_stats)):
                    if i == 0:
                        cum_irr_vals.append(valid_stats[0]['avg'])
                        cum_vol_pct.append((valid_stats[0]['area'] / valid_stats[0]['tot'] * 100) if valid_stats[0]['tot'] > 0 else 0)
                    else:
                        dz = abs(valid_stats[i]['z'] - valid_stats[i-1]['z'])
                        dV_ilum = (valid_stats[i-1]['area'] + valid_stats[i]['area']) / 2.0 * dz
                        dV_tot = (valid_stats[i-1]['tot'] + valid_stats[i]['tot']) / 2.0 * dz
                        
                        EA_prev = valid_stats[i-1]['avg'] * valid_stats[i-1]['tot']
                        EA_curr = valid_stats[i]['avg'] * valid_stats[i]['tot']
                        dEA = (EA_prev + EA_curr) / 2.0 * dz
                        
                        v_ilum_run += dV_ilum
                        v_tot_run += dV_tot
                        EA_run += dEA
                        
                        cum_irr_vals.append(EA_run / v_tot_run if v_tot_run > 0 else 0)
                        cum_vol_pct.append((v_ilum_run / v_tot_run * 100) if v_tot_run > 0 else 0)

                fig_dp, ax_dp = plt.subplots(figsize=(6, 5))
                
                irr_vals_plot = [max(val, 1e-4) for val in cum_irr_vals]
                ax_dp.plot(irr_vals_plot, z_vals, 'b-', label='Irradiancia prom. acumulada', linewidth=2.5)
                ax_dp.set_xscale('log')
                ax_dp.set_xlabel('Irradiancia promedio volumétrica [W/m²] (Log)', color='b', weight='bold')
                ax_dp.tick_params(axis='x', labelcolor='b')
                
                # Etiqueta e inversión dinámica según el modo (Jaula o Estanque)
                ax_dp.set_ylabel('Profundidad Z [m]' if env_type != 'estanque' else 'Altura desde el fondo [m]', weight='bold')
                
                ax_dp.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
                ax_dp.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
                
                ax_dp.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
                
                ax_dp.grid(True, which='major', linestyle='-', alpha=0.6)
                ax_dp.grid(True, which='minor', linestyle=':', alpha=0.3)
                
                if env_type != 'estanque':
                    ax_dp.invert_yaxis()
                
                ax_vol = ax_dp.twiny()
                ax_vol.plot(cum_vol_pct, z_vals, 'm-', label='% Vol. iluminado acumulado', linewidth=2.5)
                ax_vol.set_xlabel(f'% Volumen acumulado (>= {contour_val} W/m²)', color='m', weight='bold')
                ax_vol.tick_params(axis='x', labelcolor='m')
                ax_vol.set_xlim(-5, 105)
                
                ax_vol.xaxis.set_major_locator(ticker.MultipleLocator(10))
                
                lines_1, labels_1 = ax_dp.get_legend_handles_labels()
                lines_2, labels_2 = ax_vol.get_legend_handles_labels()
                ax_dp.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
                
                fig_dp.suptitle(f"Perfil volumétrico acumulado - Kd={kd_val} (Resolución: {profile_step}m)", fontsize=12)
                plt.tight_layout()
                
                buf_dp = io.BytesIO()
                plt.savefig(buf_dp, format='png', bbox_inches='tight')
                plt.close(fig_dp)
                kd_res["depth_profile_image"] = base64.b64encode(buf_dp.getvalue()).decode('utf-8')

            if config.get('compare_measurements') and len(comp_z) > 0:
                idx_s = np.argsort(comp_z)
                z_arr, m_arr, s_arr = np.array(comp_z)[idx_s], np.array(comp_meas)[idx_s], np.array(comp_sim)[idx_s]
                ss_res, ss_tot = np.sum((m_arr - s_arr)**2), np.sum((m_arr - np.mean(m_arr))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                rmse = np.sqrt(np.mean((m_arr - s_arr)**2))
                
                fig_comp, ax_comp = plt.subplots(figsize=(6, 5))
                ax_comp.plot(m_arr, z_arr, 'b-o', label='Medición', markersize=6, linewidth=2)
                ax_comp.plot(s_arr, z_arr, 'r--s', label='Simulación', markersize=6, linewidth=2)
                
                ax_comp.set_ylabel(r"Profundidad $Z$ $[m]$" if env_type != 'estanque' else r"Altura desde el fondo $Z$ $[m]$")
                if env_type != 'estanque':
                    ax_comp.invert_yaxis()
                    
                ax_comp.set_title(rf"Atenuación: Simulación vs Medición en $(X={config['compare_x']}, Y={config['compare_y']})$ | Kd={kd_val}")
                ax_comp.set_xlabel(r"Irradiancia $[W/m^2]$")
                ax_comp.text(0.95, 0.05, f"Métricas:\n$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}", transform=ax_comp.transAxes, fontsize=10,
                             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax_comp.grid(True, linestyle=':', alpha=0.6)
                ax_comp.legend(loc='upper right')
                
                buf_comp = io.BytesIO()
                plt.savefig(buf_comp, format='png', bbox_inches='tight')
                plt.close(fig_comp)
                kd_res["comparison_image"] = base64.b64encode(buf_comp.getvalue()).decode('utf-8')

            results_by_kd[str(kd_val)] = kd_res
            min_irr_all = 0 if min_irr_all == 999999 else min_irr_all
            power_eff = sum([float(l.get('power', 0)) * float(l.get('dim', 1.0)) for l in config['lamps']])
            lamps_str = ", ".join(list(set([l['xml'].replace('.xml','').replace('.ies', '') for l in config['lamps']])))
            pos_str = " | ".join([f"({l['x']}, {l['y']}, {l['z']})" for l in config['lamps']])
            secchi_eq = 1.7 / kd_val if kd_val > 0 else 0

            table_data.append({
                "kd": kd_val, "avg": avg_all, "max": max_irr_all, "min": min_irr_all,
                "vol_pct": vol_pct, "power_eff": power_eff, "lamps_str": lamps_str, "pos_str": pos_str, "secchi": secchi_eq
            })

        return jsonify({
            "status": "ok", 
            "depths": [str(d) for d in target_depths_requested],
            "kds": [str(k) for k in kds_requested],
            "results_by_kd": results_by_kd,
            "table_data": table_data,
            "spectrums": spectrum_results,
            "lamps_names": lamps_names
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    for f in os.listdir(UPLOAD_FOLDER):
        if f.lower().endswith('.xml') or f.lower().endswith('.ies'):
            filepath = os.path.join(UPLOAD_FOLDER, f)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file: engine.load_file(f, file.read())
            except Exception: pass
    app.run(debug=True, port=5001)