# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import matplotlib.pyplot as plt
import stim

def buildHeavyHEX(hex_rows, hex_cols):
    """building the dictionary to map coordinates (x+iy) to qubit indices a (hex_rows)x(hex_cols) lattice"""
    rows, cols = hex_rows*2+1,hex_cols*4+3

    HHXlatticepos = [[i,0,i] for i in range(cols-1)]
    i=cols-1

    row = 1
    while row < rows-1:
        col=0
        if row%2==0:
            while col < cols:
                HHXlatticepos.append([i, row, col])
                i+=1
                col+=1
        else:
            while col < cols:
                if row%4==1 and col%4==0:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                if row%4==3 and col%4==2:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                col+=1
        row+=1
    HHXlatticepos.extend([[len(HHXlatticepos)+i,rows-1,i+1] for i in range(cols-1)])
    heavyHEX_dict = {qr+qi*1j: i for i,qi,qr in HHXlatticepos}
    return heavyHEX_dict

def draw_lattice(heavyHEX_dict, coords = [], coord_col = 'red', links = [], link_col = 'red',indices:bool = True):
    """plotting function for thee heavy hex coordinate dictionary"""
    qi_list, y_list, x_list = np.transpose([[item[1],item[0].imag,item[0].real] for item in heavyHEX_dict.items()])

    plt.figure(frameon=False)

    inv_dict = {v:k for k,v in heavyHEX_dict.items()}
    for link in links:
        if len(link)==2:
            plt.plot([inv_dict[link[0]].real,inv_dict[link[1]].real],
                     [-inv_dict[link[0]].imag,-inv_dict[link[1]].imag], c=link_col)
        else:
            plt.plot([inv_dict[link[0]].real],
                     [-inv_dict[link[0]].imag],'o',c=link_col)
            
    plt.plot(x_list,-y_list,'o',c='gray',alpha = 0.65)
    for i in [int(qi) for qi in qi_list]:
        if indices:
            plt.text(x_list[i],-y_list[i],str(i),fontsize=7)

    plt.plot(np.array(coords).real,-np.array(coords).imag,'o',c=coord_col,alpha = 0.65)
    for i in [int(qi) for qi in qi_list]:
        if indices:
            plt.text(x_list[i],-y_list[i],str(i),fontsize=7)

    plt.show()
    pass

def generate_color_hex_centers(hex_dict, start_coord):
    """
        args:
            hex_dict: dictionary of the HHX lattice with the keys being the coordinates, and values are the qubit indices
            start_coord: first center coordinate either first column of first row!
        returns: coordinate list of hex centers of a given color
    """
    col_coords = []
    HHX_coords = np.array(list(hex_dict.keys()))
    # hex_around_center = np.array([-2-1j,-1j,2-1j,2+1j,1j,-2+1j])
    coord = -1j
    while coord.imag<=HHX_coords.imag.max()+2:
        while coord.real<=HHX_coords.real.max()+2:
            diff = (start_coord-coord)
            if (diff.real%12==0 and diff.imag%4==0) or (diff.real%12==6 and diff.imag%4==2):
                col_coords.append(coord)
            coord+=4
        coord = -(diff.imag%4) + coord.imag*1j+2j
    return col_coords


class Floquet_HHX():
    def __init__(self, hex_rows=18, hex_cols=9, log_obs = "X", num_cycles = 3, gate_error = 0,id_error = 0,RO_error = 0, LR_method = 0):
        """
        hex_rows: must be multiples of 2 (due to the implemented boundary conditions); X-distance (weight of log-Z) is hex_rows-1
        hex_cols: must be multiples of 3 (due to the implemented boundary conditions); Z-distance is hex_cols+1+k where k is unknown...
        num_cycles: number of full Floquet code cycles
        log_obs: logical observable to prepare (X or Z)
        gate_error, id_error, RO_error
        LR_method: 0 - no leakage reduction (LR), 1 - two rounds of LR, 2 - four rounds of LR, 3 - six rounds of LR, 4 - six round with distance reduction
        """
        self.gate_error = gate_error
        self.id_error = id_error
        self.RO_error = RO_error
        heavyHEX_dict=buildHeavyHEX(hex_rows=hex_rows,hex_cols=hex_cols)
        self.heavyHEX_dict = heavyHEX_dict
        self.log_obs = log_obs

        heavyHEX_dict_inv={v:k for k,v in heavyHEX_dict.items()}
        HHX_coords = np.array(list(heavyHEX_dict.keys()))
        excluded_edge_HEX_center = [-1j, -1j+HHX_coords.real.max()+2, 1j+HHX_coords.real.max(),
                                    HHX_coords.imag.max()*1j-1j, -2+HHX_coords.imag.max()*1j+1j, HHX_coords.real.max()+HHX_coords.imag.max()*1j+1j]

        pair_pos_list_a = [[-2-1j,-1j],
                        [2-1j,2+1j],
                        [-2+1j,1j]
                        ] 
        pair_pos_list_b = [[-1j, 2-1j],
                        [2+1j,1j],
                        [-2+1j,-2-1j]
                        ]

        anc_pos_list_a = [-1-1j,2,-1+1j]
        anc_pos_list_b = [1-1j,1+1j,-2]

        # links around blue hexagons
        blue_centers = generate_color_hex_centers(hex_dict=heavyHEX_dict, start_coord=2+1j)
        green_links  = {}
        red_links = {}
        for center in blue_centers:
            for rel_anc_pos, rel_pair_pos in zip(anc_pos_list_a,pair_pos_list_a):
                qubits = [heavyHEX_dict[center+rel_pos] for rel_pos in rel_pair_pos if center+rel_pos in heavyHEX_dict]
                if len(qubits)!=0:
                    green_links[center+rel_anc_pos] = qubits
            for rel_anc_pos, rel_pair_pos in zip(anc_pos_list_b,pair_pos_list_b):
                qubits = [heavyHEX_dict[center+rel_pos] for rel_pos in rel_pair_pos if center+rel_pos in heavyHEX_dict]
                if len(qubits)!=0:
                    red_links[center+rel_anc_pos] = qubits
        blue_centers = [coord for coord in blue_centers 
                        if coord not in excluded_edge_HEX_center] #restricting blue_centers to the measured plaquettes
        blue_hex_green_link_coords = self.get_link_coords(centers=blue_centers, anc_pos_list=anc_pos_list_a, pair_pos_list=pair_pos_list_a)
        blue_hex_red_link_coords = self.get_link_coords(centers=blue_centers, anc_pos_list=anc_pos_list_b, pair_pos_list=pair_pos_list_b)

        # links around red hexagons
        red_centers = generate_color_hex_centers(hex_dict=heavyHEX_dict, start_coord=6+1j)
        blue_links = {}
        for center in red_centers:
            for rel_anc_pos, rel_pair_pos in zip(anc_pos_list_a,pair_pos_list_a):
                qubits = [heavyHEX_dict[center+rel_pos] for rel_pos in rel_pair_pos if center+rel_pos in heavyHEX_dict]
                if len(qubits)!=0:
                    blue_links[center+rel_anc_pos] =  qubits

        red_centers = [coord for coord in red_centers 
                        if coord not in excluded_edge_HEX_center] #restricting red_centers to the measured plaquettes
        
        red_hex_blue_link_coords = self.get_link_coords(centers=red_centers, anc_pos_list=anc_pos_list_a, pair_pos_list=pair_pos_list_a)
        red_hex_green_link_coords = self.get_link_coords(centers=red_centers, anc_pos_list=anc_pos_list_b, pair_pos_list=pair_pos_list_b)

        # links around green hexagons
        green_centers = generate_color_hex_centers(hex_dict=heavyHEX_dict, start_coord=-2+1j)
        green_centers = [coord for coord in green_centers 
                        if coord not in excluded_edge_HEX_center]
        green_hex_red_link_coords = self.get_link_coords(centers=green_centers, anc_pos_list=anc_pos_list_a, pair_pos_list=pair_pos_list_a)
        green_hex_blue_link_coords = self.get_link_coords(centers=green_centers, anc_pos_list=anc_pos_list_b, pair_pos_list=pair_pos_list_b)
    
        self.data_qubits = [q for coord,q in heavyHEX_dict.items() if coord.real%2==0 and coord.imag%2==0]

        max_coord_real = np.array(list(heavyHEX_dict.keys())).real.max()
        max_coord_imag = np.array(list(heavyHEX_dict.keys())).imag.max()

        self.stim_circuit = stim.Circuit()

        for q_coord, q_ind in heavyHEX_dict.items():
            self.stim_circuit.append("QUBIT_COORDS",q_ind,[q_coord.real,q_coord.imag])

        round = 0
        self.measure_dict = {}
        max_round_meas_ind = 0

        self.stim_circuit.append("R",self.data_qubits)
        self.stim_circuit.append("X_ERROR",self.data_qubits,self.RO_error)
        if log_obs == "X":
            self.stim_circuit.append("H",self.data_qubits)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,self.gate_error)
        

        for _ in range(num_cycles):
            links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in red_links.items()
                      if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord.imag%2==0] #horizontal links LRU possible
            links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in red_links.items() 
                      if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord.imag%2==1] #vertical link LRU reduces code distance
            if 4>LR_method>0:
                self.link_measure_LR(basis="X",link_dicts=links1, round=round)
            else:
                self.link_measure(basis="X",link_dicts=links1, round=round)
            if LR_method>3:
                self.link_measure_LR(basis="X",link_dicts=links1+links2, round=round, tick = True)            
            else:
                self.link_measure(basis="X",link_dicts=links2, round=round)
            self.stim_circuit.append("TICK")
            links = links1+links2
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links]
            ### include every red link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = green_hex_red_link_coords,
                              hex_start_list = green_hex_blue_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            round+=1

            links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in green_links.items()
                      if anc_coord.imag%2==1] #vertical links LRU possible
            links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in green_links.items() 
                      if anc_coord.imag%2==0] #horizontal link LRU reduces code distance
            if 4>LR_method>1:
                self.link_measure_LR(basis="Z",link_dicts=links1, round=round)
            else:
                self.link_measure(basis="Z",link_dicts=links1, round=round)
            if LR_method>3:
                self.link_measure_LR(basis="Z",link_dicts=links1+links2, round=round, tick = True)            
            else:
                self.link_measure(basis="Z",link_dicts=links2, round=round)
            self.stim_circuit.append("TICK")
            links = links1+links2
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links if dic['anc_coord'].real!=max_coord_real-1]
            ### include every green link measurement with coord.real==4 in the observable
            obs_links = [dic['anc_coord'] for dic in links if dic['anc_coord'].real==4]
            self.add_dets_obs(hex_fin_list = blue_hex_green_link_coords,
                              hex_start_list = blue_hex_red_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            round+=1

            links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in blue_links.items()
                      if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord.imag%2==0] #horizontal links LRU possible
            links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in blue_links.items() 
                      if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord.imag%2==1] #vertical link LRU reduces code distance
            if 4>LR_method>0:
                self.link_measure_LR(basis="X",link_dicts=links1, round=round)
            else:
                self.link_measure(basis="X",link_dicts=links1, round=round)
            if LR_method>3:
                self.link_measure_LR(basis="X",link_dicts=links1+links2, round=round, tick = True)            
            else:
                self.link_measure(basis="X",link_dicts=links2, round=round)
            self.stim_circuit.append("TICK")
            links = links1 + links2
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links if dic['anc_coord'].imag not in [0, max_coord_imag]]
            ### include every blue link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = red_hex_blue_link_coords,
                              hex_start_list = red_hex_green_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            round+=1

            links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in red_links.items()
                      if anc_coord.real != max_coord_real+1 and anc_coord.imag%2==1] #vertical links LRU possible
            links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in red_links.items() 
                      if anc_coord.real != max_coord_real+1 and anc_coord.imag%2==0] #horizontal link LRU reduces code distance
            if 4>LR_method>1:
                self.link_measure_LR(basis="Z",link_dicts=links1, round=round)
            else:
                self.link_measure(basis="Z",link_dicts=links1, round=round)
            if LR_method>3:
                self.link_measure_LR(basis="Z",link_dicts=links1+links2, round=round, tick = True)            
            else:
                self.link_measure(basis="Z",link_dicts=links2, round=round)
            self.stim_circuit.append("TICK")
            links = links1+links2
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links if dic['anc_coord'].real!=0]
            ### include every red link measurement with coord.real==6 in the observable
            obs_links = [dic['anc_coord'] for dic in links if dic['anc_coord'].real==6]
            self.add_dets_obs(hex_fin_list = green_hex_red_link_coords,
                              hex_start_list = green_hex_blue_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            round+=1

            links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in green_links.items()
                      if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord not in [1+1j*max_coord_imag, max_coord_real-1] and anc_coord.imag%2==0] #horizontal links LRU possible
            links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in green_links.items() 
                      if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord not in [1+1j*max_coord_imag, max_coord_real-1] and anc_coord.imag%2==1] #vertical link LRU reduces code distance
            if 4>LR_method>2:
                self.link_measure_LR(basis="X",link_dicts=links1, round=round)
            else:
                self.link_measure(basis="X",link_dicts=links1, round=round)
            if LR_method>3:
                self.link_measure_LR(basis="X",link_dicts=links1+links2, round=round, tick = True)            
            else:
                self.link_measure(basis="X",link_dicts=links2, round=round)
            self.stim_circuit.append("TICK")
            links = links1 + links2 
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links]
            ### include every green link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = blue_hex_green_link_coords,
                              hex_start_list = blue_hex_red_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            round+=1

            links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in blue_links.items()
                      if anc_coord.real != -1 and anc_coord.imag%2==1] #vertical links LRU possible
            links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in blue_links.items() 
                      if anc_coord.real != -1 and anc_coord.imag%2==0] #horizontal link LRU reduces code distance
            if 4>LR_method>2:
                self.link_measure_LR(basis="Z",link_dicts=links1, round=round)
            else:
                self.link_measure(basis="Z",link_dicts=links1, round=round)
            if LR_method>3:
                self.link_measure_LR(basis="Z",link_dicts=links1+links2, round=round, tick = True)            
            else:
                self.link_measure(basis="Z",link_dicts=links2, round=round)
            self.stim_circuit.append("TICK")
            links = links1+links2
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links]
            ## include every blue link measurement with coord.real==5 in the observable
            obs_links = [dic['anc_coord'] for dic in links if dic['anc_coord'].real==5]
            self.add_dets_obs(hex_fin_list = red_hex_blue_link_coords,
                              hex_start_list = red_hex_green_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            round+=1

        ##### final measurements
        max_round_meas_ind = self.stim_circuit.num_measurements
        if log_obs == "X":
            self.stim_circuit.append("H",self.data_qubits)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,gate_error)
            self.stim_circuit.append("X_ERROR",self.data_qubits,RO_error)
            self.stim_circuit.append("M",self.data_qubits)
            self.measure_dict.update({(heavyHEX_dict_inv[q],round):[max_round_meas_ind+i] for i,q in enumerate(self.data_qubits)})
            max_round_meas_ind = self.stim_circuit.num_measurements
            for hex in green_hex_blue_link_coords:
                if all([anc.imag not in [0, max_coord_imag] for anc in hex]): #top/bottom rows are only Z detectors
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in blue_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-4)]]])
            for hex in red_hex_green_link_coords:
                if all([anc.imag not in [0, max_coord_imag] for anc in hex]): #top/bottom rows are only Z detectors
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in green_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-2)]]])
                    
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(pos+2j,round)][0] for pos in range(0,int(max_coord_real)+2,2) if pos%6!=2]],0)        
        
        elif log_obs == "Z":
            self.stim_circuit.append("X_ERROR",self.data_qubits,RO_error)
            self.stim_circuit.append("M",self.data_qubits)
            self.measure_dict.update({(heavyHEX_dict_inv[q],round):[max_round_meas_ind+i] for i,q in enumerate(self.data_qubits)})
            max_round_meas_ind = self.stim_circuit.num_measurements

            for hex in green_hex_blue_link_coords:
                if all([anc.real != -1 for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in blue_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-1)]]])
            for hex in blue_hex_red_link_coords:
                if all([anc.real != max_coord_real-2 for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in red_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-3)]]])

            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(6 + pos*1j,round)][0] for pos in range(0,int(max_coord_imag)+2,2)]],0)
        
        self.d = len(self.stim_circuit.shortest_graphlike_error())    

    def get_link_coords(self,centers,anc_pos_list,pair_pos_list):
        link_coords = [
            [center+rel_anc_pos 
                for rel_anc_pos, rel_pair_pos in zip(anc_pos_list,pair_pos_list)
                if center+rel_pair_pos[0] in self.heavyHEX_dict or center+rel_pair_pos[1] in self.heavyHEX_dict
            ]
            for center in centers 
            ]
        return link_coords

    def link_measure(self, basis,link_dicts, round):
        max_meas_ind = self.stim_circuit.num_measurements
        for link_dict in link_dicts:
            qubits = link_dict['qubits']
            if len(qubits)==2:
                ancilla = self.heavyHEX_dict[link_dict['anc_coord']]
                self.stim_circuit.append("R",ancilla)
                # self.stim_circuit.append("TICK")
                self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                if basis=="X":
                    self.stim_circuit.append("H",ancilla)
                    self.stim_circuit.append("DEPOLARIZE1",ancilla,self.gate_error)
                    self.stim_circuit.append("CX",[ancilla, qubits[0]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[ancilla, qubits[1]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    self.stim_circuit.append("H",ancilla)
                    self.stim_circuit.append("DEPOLARIZE1",ancilla,self.gate_error)
                elif basis=="Z":
                    self.stim_circuit.append("CX",[qubits[0],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[qubits[1],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                # self.stim_circuit.append("TICK")
                self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                self.stim_circuit.append("M",ancilla)
                # self.stim_circuit.append("TICK")
                self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind]})
                max_meas_ind+=1
            else:
                if basis=="X":
                    self.stim_circuit.append("H",qubits[0])
                    self.stim_circuit.append("DEPOLARIZE1",qubits[0],self.gate_error)
                self.stim_circuit.append("X_ERROR",qubits[0],self.RO_error)
                self.stim_circuit.append("M",qubits[0])
                # self.stim_circuit.append("TICK")
                self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind]})
                max_meas_ind+=1
                if basis=="X":
                    self.stim_circuit.append("H",qubits[0])
                    self.stim_circuit.append("DEPOLARIZE1",qubits[0],self.gate_error)
        # self.stim_circuit.append("TICK")


    def link_measure_LR(self, basis,link_dicts, round, tick = False):
        max_meas_ind = self.stim_circuit.num_measurements
        for link_dict in link_dicts:
            qubits = link_dict['qubits']
            if len(qubits)==2:
                ancilla = self.heavyHEX_dict[link_dict['anc_coord']]
                self.stim_circuit.append("R",ancilla)
                # self.stim_circuit.append("TICK")
                self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                if basis=="X":
                    self.stim_circuit.append("CX",[qubits[0],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[qubits[1],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    # self.stim_circuit.append("TICK")

                    self.stim_circuit.append("H",qubits)
                    self.stim_circuit.append("DEPOLARIZE1",qubits,self.gate_error)
                    self.stim_circuit.append("X_ERROR",qubits,self.RO_error)
                    self.stim_circuit.append("M",qubits)
                    self.stim_circuit.append("H",qubits)
                    self.stim_circuit.append("DEPOLARIZE1",qubits,self.gate_error)
                    # self.stim_circuit.append("TICK")
                    self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind,max_meas_ind+1]})
                    max_meas_ind+=2

                elif basis=="Z":
                    self.stim_circuit.append("H",ancilla)
                    self.stim_circuit.append("DEPOLARIZE1",ancilla,self.gate_error)
                    # self.stim_circuit.append("TICK")

                    self.stim_circuit.append("CX",[ancilla,qubits[0]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[ancilla,qubits[1]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    # self.stim_circuit.append("TICK")

                    self.stim_circuit.append("X_ERROR",qubits,self.RO_error)
                    self.stim_circuit.append("M",qubits)
                    self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind,max_meas_ind+1]})
                    # self.stim_circuit.append("TICK")
                    max_meas_ind+=2

            else:
                if basis=="X":
                    self.stim_circuit.append("H",qubits[0])
                    self.stim_circuit.append("DEPOLARIZE1",qubits[0],self.gate_error)
                self.stim_circuit.append("X_ERROR",qubits[0],self.RO_error)
                self.stim_circuit.append("M",qubits[0])
                # self.stim_circuit.append("TICK")
                self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind]})
                max_meas_ind+=1
                if basis=="X":
                    self.stim_circuit.append("H",qubits[0])
                    self.stim_circuit.append("DEPOLARIZE1",qubits[0],self.gate_error)

        if tick:    
            self.stim_circuit.append("TICK")

        for link_dict in link_dicts:
            qubits = link_dict['qubits']
            if len(qubits)==2:
                ancilla = self.heavyHEX_dict[link_dict['anc_coord']]
                if basis=="X":


                    self.stim_circuit.append("CX",[qubits[0],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[qubits[1],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    # self.stim_circuit.append("TICK")

                    self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                    self.stim_circuit.append("M",ancilla)
                    self.stim_circuit.append("CX",[stim.target_rec(-1),qubits[0]])
                    # self.stim_circuit.append("TICK")
                    max_meas_ind+=1
                elif basis=="Z":


                    self.stim_circuit.append("CX",[ancilla,qubits[0]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[ancilla,qubits[1]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    # self.stim_circuit.append("TICK")

                    self.stim_circuit.append("H",ancilla)
                    self.stim_circuit.append("DEPOLARIZE1",ancilla,self.gate_error)
                    self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                    self.stim_circuit.append("M",ancilla)
                    # self.stim_circuit.append("TICK")

                    self.stim_circuit.append("CZ",[stim.target_rec(-1),qubits[0]])
                    max_meas_ind+=1


    def add_dets_obs(self,hex_fin_list,hex_start_list,basis,obs_links,ancillas,round):
        max_round_meas_ind = self.stim_circuit.num_measurements
        for hex, hex_start in zip(hex_fin_list, hex_start_list):
            if all([anc in ancillas for anc in hex]):
                if self.log_obs == basis and round<5:
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round)]]])
                elif round>=5:
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round)]]]+
                                                            [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex_start for target in self.measure_dict[(anc,round-4)]]])
        if self.log_obs == basis and round!=0: #having this update in round-0 reduces the code distance...
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in obs_links for target in self.measure_dict[(anc,round)]]],0)


class Double_Floquet_HHX():
    def __init__(self, hex_rows=6, hex_cols=3, log_obs = "X", num_cycles = 3, gate_error = 0,id_error = 0,RO_error = 0, dangling_bonds = False):
        """
        hex_rows: 2*hex_rows is the # of hexagons in a row (due to the implemented boundary conditions);
        hex_cols: 3*hex_cols is the # of hexagons in a column (due to the implemented boundary conditions);
        num_cycles: number of full Floquet code cycles
        log_obs: logical observable to prepare (X or Z)
        gate_error, id_error, RO_error
        dangling_bonds: whether to add extra sites to avoid collisions between the schedule of the two floquet codes at the boundary
        """
        self.gate_error = gate_error
        self.id_error = id_error
        self.RO_error = RO_error
        self.log_obs = log_obs
        self.dangling_bonds = dangling_bonds

        heavyHEX_dict=buildHeavyHEX(hex_rows=hex_rows,hex_cols=hex_cols)
        HHX_coords = np.array(list(heavyHEX_dict.keys()))
        max_coord_real = HHX_coords.real.max()
        max_coord_imag = HHX_coords.imag.max()

        excluded_edge_HEX_center = [-1j, -1j+HHX_coords.real.max()+2, 1j+HHX_coords.real.max(),
                                    HHX_coords.imag.max()*1j-1j, -2+HHX_coords.imag.max()*1j+1j, HHX_coords.real.max()+HHX_coords.imag.max()*1j+1j]

        self.data_qubits = [q for coord,q in heavyHEX_dict.items() if coord.real%2==0 and coord.imag%2==0]
        max_qind = max(self.data_qubits)
        boundary_swap_pairs = [
            heavyHEX_dict[max_coord_real-1],heavyHEX_dict[max_coord_real-2], #corner qubit with existing ancillas
            heavyHEX_dict[1+max_coord_imag*1j],heavyHEX_dict[2+max_coord_imag*1j] #corner qubit with existing ancillas
            ]
        if dangling_bonds:
            for coord in HHX_coords:
                if coord.imag == 0 and coord.real%4==2:
                    heavyHEX_dict[coord-1j]=max_qind+1
                    boundary_swap_pairs+=[heavyHEX_dict[coord],max_qind+1]
                    max_qind+=1
                elif coord.real == max_coord_real and coord.imag%2==0:
                    heavyHEX_dict[coord+1]=max_qind+1
                    boundary_swap_pairs+=[heavyHEX_dict[coord],max_qind+1]
                    max_qind+=1
                elif coord.imag == max_coord_imag and coord.real%4==0:
                    heavyHEX_dict[coord+1j]=max_qind+1
                    boundary_swap_pairs+=[heavyHEX_dict[coord],max_qind+1]
                    max_qind+=1
                elif coord.real == 0 and coord.imag%2==0:
                    heavyHEX_dict[coord-1]=max_qind+1
                    boundary_swap_pairs+=[heavyHEX_dict[coord],max_qind+1]
                    max_qind+=1
        
        heavyHEX_dict_inv={v:k for k,v in heavyHEX_dict.items()}
        self.heavyHEX_dict = heavyHEX_dict



        pair_pos_list_a = [[-2-1j,-1j],
                        [2-1j,2+1j],
                        [-2+1j,1j]
                        ] 
        pair_pos_list_b = [[-1j, 2-1j],
                        [2+1j,1j],
                        [-2+1j,-2-1j]
                        ]

        anc_pos_list_a = [-1-1j,2,-1+1j]
        anc_pos_list_b = [1-1j,1+1j,-2]

        # links around blue hexagons
        blue_centers = generate_color_hex_centers(hex_dict=heavyHEX_dict, start_coord=2+1j)
        green_links  = {}
        red_links = {}
        for center in blue_centers:
            for rel_anc_pos, rel_pair_pos in zip(anc_pos_list_a,pair_pos_list_a):
                qubits = [heavyHEX_dict[center+rel_pos] for rel_pos in rel_pair_pos if center+rel_pos in heavyHEX_dict]
                if len(qubits)!=0:
                    green_links[center+rel_anc_pos] = qubits
            for rel_anc_pos, rel_pair_pos in zip(anc_pos_list_b,pair_pos_list_b):
                qubits = [heavyHEX_dict[center+rel_pos] for rel_pos in rel_pair_pos if center+rel_pos in heavyHEX_dict]
                if len(qubits)!=0:
                    red_links[center+rel_anc_pos] = qubits
        blue_centers = [coord for coord in blue_centers 
                        if coord not in excluded_edge_HEX_center] #restricting blue_centers to the measured plaquettes
        blue_hex_green_link_coords = self.get_link_coords(centers=blue_centers, anc_pos_list=anc_pos_list_a, pair_pos_list=pair_pos_list_a)
        blue_hex_red_link_coords = self.get_link_coords(centers=blue_centers, anc_pos_list=anc_pos_list_b, pair_pos_list=pair_pos_list_b)

        # links around red hexagons
        red_centers = generate_color_hex_centers(hex_dict=heavyHEX_dict, start_coord=6+1j)
        blue_links = {}
        for center in red_centers:
            for rel_anc_pos, rel_pair_pos in zip(anc_pos_list_a,pair_pos_list_a):
                qubits = [heavyHEX_dict[center+rel_pos] for rel_pos in rel_pair_pos if center+rel_pos in heavyHEX_dict]
                if len(qubits)!=0:
                    blue_links[center+rel_anc_pos] =  qubits

        red_centers = [coord for coord in red_centers 
                        if coord not in excluded_edge_HEX_center] #restricting red_centers to the measured plaquettes
        
        red_hex_blue_link_coords = self.get_link_coords(centers=red_centers, anc_pos_list=anc_pos_list_a, pair_pos_list=pair_pos_list_a)
        red_hex_green_link_coords = self.get_link_coords(centers=red_centers, anc_pos_list=anc_pos_list_b, pair_pos_list=pair_pos_list_b)

        # links around green hexagons
        green_centers = generate_color_hex_centers(hex_dict=heavyHEX_dict, start_coord=-2+1j)
        green_centers = [coord for coord in green_centers 
                        if coord not in excluded_edge_HEX_center]
        green_hex_red_link_coords = self.get_link_coords(centers=green_centers, anc_pos_list=anc_pos_list_a, pair_pos_list=pair_pos_list_a)
        green_hex_blue_link_coords = self.get_link_coords(centers=green_centers, anc_pos_list=anc_pos_list_b, pair_pos_list=pair_pos_list_b)

        self.stim_circuit = stim.Circuit()

        for q_coord, q_ind in heavyHEX_dict.items():
            self.stim_circuit.append("QUBIT_COORDS",q_ind,[q_coord.real,q_coord.imag])

        round = 0
        self.measure_dict = {}
        max_round_meas_ind = 0

        ###########logical init##################

        every_qubit = heavyHEX_dict.values()

        self.stim_circuit.append("R",every_qubit)
        self.stim_circuit.append("X_ERROR",every_qubit,self.RO_error)
        if log_obs=="X":
            self.stim_circuit.append("H",every_qubit)
            self.stim_circuit.append("DEPOLARIZE1",every_qubit,self.RO_error)


        links0 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in red_links.items()
                    if anc_coord.imag not in [-1, max_coord_imag+1]]
        links1 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in green_links.items()]
        links2 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in blue_links.items()
                    if anc_coord.imag not in [-1, max_coord_imag+1]]
        links3 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in red_links.items()
                    if anc_coord.real != max_coord_real+1]
        links4 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in green_links.items()
                    if anc_coord.imag not in [-1, max_coord_imag+1] and anc_coord not in [1+1j*max_coord_imag, max_coord_real-1]]
        links5 = [{'qubits':qubits,'anc_coord': anc_coord} for anc_coord,qubits in blue_links.items()
                    if anc_coord.real != -1]

        for cycle in range(num_cycles):
            self.link_expand(basis="Z",link_dicts=links5, round=round-1)
            self.link_contract(basis="X",link_dicts=links0, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links0]
            ### include every red link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links0 if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = green_hex_red_link_coords,
                              hex_start_list = green_hex_blue_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2
            self.link_expand(basis="Z",link_dicts=links1, round=round-1)
            self.link_contract(basis="X",link_dicts=links2, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links2 if dic['anc_coord'].imag not in [0, max_coord_imag]]
            ### include every blue link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links2 if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = red_hex_blue_link_coords,
                              hex_start_list = red_hex_green_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2

            
            self.link_expand(basis="X",link_dicts=links0, round=round-1)
            self.link_contract(basis="Z",link_dicts=links1, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links1 if dic['anc_coord'].real!=max_coord_real-1]
            ### include every green link measurement with coord.real==4 in the observable
            obs_links = [dic['anc_coord'] for dic in links1 if dic['anc_coord'].real==4]
            self.add_dets_obs(hex_fin_list = blue_hex_green_link_coords,
                              hex_start_list = blue_hex_red_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2
            self.link_expand(basis="X",link_dicts=links2, round=round-1)
            self.link_contract(basis="Z",link_dicts=links3, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links3 if dic['anc_coord'].real!=0]
            ### include every red link measurement with coord.real==6 in the observable
            obs_links = [dic['anc_coord'] for dic in links3 if dic['anc_coord'].real==6]
            self.add_dets_obs(hex_fin_list = green_hex_red_link_coords,
                              hex_start_list = green_hex_blue_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2


            self.link_expand(basis="Z",link_dicts=links1, round=round-1)
            self.link_contract(basis="X",link_dicts=links2, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links2 if dic['anc_coord'].imag not in [0, max_coord_imag]]
            ### include every blue link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links2 if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = red_hex_blue_link_coords,
                              hex_start_list = red_hex_green_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2
            self.link_expand(basis="Z",link_dicts=links3, round=round-1)
            self.link_contract(basis="X",link_dicts=links4, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links4]
            ### include every green link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links4 if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = blue_hex_green_link_coords,
                              hex_start_list = blue_hex_red_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2


            self.link_expand(basis="X",link_dicts=links2, round=round-1)
            self.link_contract(basis="Z",link_dicts=links3, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links3 if dic['anc_coord'].real!=0]
            ### include every red link measurement with coord.real==6 in the observable
            obs_links = [dic['anc_coord'] for dic in links3 if dic['anc_coord'].real==6]
            self.add_dets_obs(hex_fin_list = green_hex_red_link_coords,
                              hex_start_list = green_hex_blue_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2
            self.link_expand(basis="X",link_dicts=links4, round=round-1)
            self.link_contract(basis="Z",link_dicts=links5, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links5]
            ## include every blue link measurement with coord.real==5 in the observable
            obs_links = [dic['anc_coord'] for dic in links5 if dic['anc_coord'].real==5]
            self.add_dets_obs(hex_fin_list = red_hex_blue_link_coords,
                              hex_start_list = red_hex_green_link_coords,
                              basis = "Z",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2


            self.link_expand(basis="Z",link_dicts=links3, round=round-1)
            self.link_contract(basis="X",link_dicts=links4, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links4]
            ### include every green link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links4 if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = blue_hex_green_link_coords,
                              hex_start_list = blue_hex_red_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2
            self.link_expand(basis="Z",link_dicts=links5, round=round-1)
            self.link_contract(basis="X",link_dicts=links0, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links0]
            ### include every red link measurement with coord.imag==2 in the observable
            obs_links = [dic['anc_coord'] for dic in links0 if dic['anc_coord'].imag==2]
            self.add_dets_obs(hex_fin_list = green_hex_red_link_coords,
                              hex_start_list = green_hex_blue_link_coords,
                              basis = "X",
                              obs_links=obs_links,
                              ancillas = ancillas,
                              round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2


            self.link_expand(basis="X",link_dicts=links4, round=round-1)
            self.link_contract(basis="Z",link_dicts=links5, round=round)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
            ancillas = [dic['anc_coord'] for dic in links5]
            ## include every blue link measurement with coord.real==5 in the observable
            obs_links = [dic['anc_coord'] for dic in links5 if dic['anc_coord'].real==5]
            self.add_dets_obs(hex_fin_list = red_hex_blue_link_coords,
                            hex_start_list = red_hex_green_link_coords,
                            basis = "Z",
                            obs_links=obs_links,
                            ancillas = ancillas,
                            round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2
            if cycle<num_cycles-1:
                self.link_expand(basis="X",link_dicts=links0, round=round-1)
                self.link_contract(basis="Z",link_dicts=links1, round=round)
                self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,id_error)
                ancillas = [dic['anc_coord'] for dic in links1 if dic['anc_coord'].real!=max_coord_real-1]
                ### include every green link measurement with coord.real==4 in the observable
                obs_links = [dic['anc_coord'] for dic in links1 if dic['anc_coord'].real==4]
                self.add_dets_obs(hex_fin_list = blue_hex_green_link_coords,
                                hex_start_list = blue_hex_red_link_coords,
                                basis = "Z",
                                obs_links=obs_links,
                                ancillas = ancillas,
                                round = round)
            if dangling_bonds:
                self.stim_circuit.append("SWAP", boundary_swap_pairs)
            round+=1/2

        self.link_expand(basis="Z",link_dicts=links5, round=round-1)

        ##### final measurements LQ1
        max_round_meas_ind = self.stim_circuit.num_measurements
        if log_obs == "X":
            self.stim_circuit.append("H",self.data_qubits)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,gate_error)
            self.stim_circuit.append("X_ERROR",self.data_qubits,RO_error)
            self.stim_circuit.append("M",self.data_qubits)
            self.measure_dict.update({(heavyHEX_dict_inv[q],round):[max_round_meas_ind+i] for i,q in enumerate(self.data_qubits)})
            max_round_meas_ind = self.stim_circuit.num_measurements
            for hex in green_hex_blue_link_coords:
                if all([anc.imag not in [0, max_coord_imag] for anc in hex]): #top/bottom rows are only Z detectors
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in blue_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-4)]]])
            for hex in red_hex_green_link_coords:
                if all([anc.imag not in [0, max_coord_imag] for anc in hex]): #top/bottom rows are only Z detectors
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in green_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-2)]]])
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(pos+2j,round)][0] for pos in range(0,int(max_coord_real)+2,2) if pos%6!=2]],0)        
        
        elif log_obs == "Z":
            self.stim_circuit.append("X_ERROR",self.data_qubits,RO_error)
            self.stim_circuit.append("M",self.data_qubits)
            self.measure_dict.update({(heavyHEX_dict_inv[q],round):[max_round_meas_ind+i] for i,q in enumerate(self.data_qubits)})
            max_round_meas_ind = self.stim_circuit.num_measurements

            for hex in green_hex_blue_link_coords:
                if all([anc.real != -1 for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in blue_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-1)]]])
            for hex in blue_hex_red_link_coords:
                if all([anc.real != max_coord_real-2 for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in red_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-3)]]])
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(6 + pos*1j,round)][0] for pos in range(0,int(max_coord_imag)+2,2)]],0)

        ##### final measurements LQ2
        round+=1/2
        self.stim_circuit.append("R",self.data_qubits)
        if dangling_bonds:
            self.stim_circuit.append("SWAP", boundary_swap_pairs)
        self.link_expand(basis="X",link_dicts=links0, round=round-2)

        max_round_meas_ind = self.stim_circuit.num_measurements
        if log_obs == "X":
            self.stim_circuit.append("H",self.data_qubits)
            self.stim_circuit.append("DEPOLARIZE1",self.data_qubits,gate_error)
            self.stim_circuit.append("X_ERROR",self.data_qubits,RO_error)
            self.stim_circuit.append("M",self.data_qubits)
            self.measure_dict.update({(heavyHEX_dict_inv[q],round):[max_round_meas_ind+i] for i,q in enumerate(self.data_qubits)})
            max_round_meas_ind = self.stim_circuit.num_measurements
            for hex in blue_hex_red_link_coords:
                if all([anc.imag not in [-1, max_coord_imag+1] for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in red_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-2)]]])
            for hex in red_hex_green_link_coords:
                if all([anc.imag not in [-1, max_coord_imag+1] and anc not in [1+1j*max_coord_imag, max_coord_real-1] for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in green_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-4)]]])
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(x+2j,round)][0] for x in range(0,int(max_coord_real)+2,2) if x%6!=4]],1)

        elif log_obs == "Z":
            self.stim_circuit.append("X_ERROR",self.data_qubits,RO_error)
            self.stim_circuit.append("M",self.data_qubits)
            self.measure_dict.update({(heavyHEX_dict_inv[q],round):[max_round_meas_ind+i] for i,q in enumerate(self.data_qubits)})
            max_round_meas_ind = self.stim_circuit.num_measurements

            for hex in green_hex_blue_link_coords:
                if all([anc.real not in [-1,max_coord_real] for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in blue_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-3)]]])
            for hex in blue_hex_red_link_coords:
                if all([anc.real != max_coord_real-2 for anc in hex]):
                    self.stim_circuit.append("DETECTOR", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(heavyHEX_dict_inv[qubit],round)][0] for anc in hex for qubit in red_links[anc]]]+
                                                         [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round-5)]]])
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [self.measure_dict[(6 +y*1j,round)][0] for y in range(0,int(max_coord_imag)+2,2)]],1)

        self.d = len(self.stim_circuit.shortest_graphlike_error())    

    def get_link_coords(self,centers,anc_pos_list,pair_pos_list):
        link_coords = [
            [center+rel_anc_pos 
                for rel_anc_pos, rel_pair_pos in zip(anc_pos_list,pair_pos_list)
                if center+rel_pair_pos[0] in self.heavyHEX_dict or center+rel_pair_pos[1] in self.heavyHEX_dict
            ]
            for center in centers 
            ]
        return link_coords


    def link_measure_LR(self, basis,link_dicts, round):
        self.link_contract(basis,link_dicts, round)
        self.link_expand(basis,link_dicts, round)

    def link_contract(self, basis,link_dicts, round):
        max_meas_ind = self.stim_circuit.num_measurements
        for link_dict in link_dicts:
            qubits = link_dict['qubits']
            if len(qubits)==2:
                ancilla = self.heavyHEX_dict[link_dict['anc_coord']]
                self.stim_circuit.append("R",ancilla)
                self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                if basis=="X":
                    self.stim_circuit.append("CX",[qubits[0],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[qubits[1],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    self.stim_circuit.append("H",qubits)
                    self.stim_circuit.append("DEPOLARIZE1",qubits,self.gate_error)
                    self.stim_circuit.append("X_ERROR",qubits,self.RO_error)
                    self.stim_circuit.append("M",qubits)
                    self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind,max_meas_ind+1]})
                    max_meas_ind+=2
                    self.stim_circuit.append("R",qubits)
                    self.stim_circuit.append("X_ERROR",qubits,self.RO_error)

                elif basis=="Z":
                    self.stim_circuit.append("H",ancilla)
                    self.stim_circuit.append("DEPOLARIZE1",ancilla,self.gate_error)
                    self.stim_circuit.append("CX",[ancilla,qubits[0]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[ancilla,qubits[1]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    self.stim_circuit.append("X_ERROR",qubits,self.RO_error)
                    self.stim_circuit.append("M",qubits)
                    self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind,max_meas_ind+1]})
                    max_meas_ind+=2
                    self.stim_circuit.append("R",qubits)
                    self.stim_circuit.append("X_ERROR",qubits,self.RO_error)
            else:
                if basis=="X":
                    self.stim_circuit.append("H",qubits[0])
                    self.stim_circuit.append("DEPOLARIZE1",qubits[0],self.gate_error)
                self.stim_circuit.append("X_ERROR",qubits[0],self.RO_error)
                self.stim_circuit.append("M",qubits[0])
                self.measure_dict.update({(link_dict['anc_coord'],round):[max_meas_ind]})
                max_meas_ind+=1
                self.stim_circuit.append("R",qubits[0])
                self.stim_circuit.append("X_ERROR",qubits,self.RO_error)
        self.stim_circuit.append("TICK")

    def link_expand(self, basis,link_dicts, round):
        max_meas_ind = self.stim_circuit.num_measurements
        for link_dict in link_dicts:
            qubits = link_dict['qubits']
            if len(qubits)==2:
                ancilla = self.heavyHEX_dict[link_dict['anc_coord']]
                if basis=="X":
                    if round>=0:
                        # qubits are reset
                        meas_ind0,meas_ind1 = self.measure_dict[(link_dict['anc_coord'],round)]
                        self.stim_circuit.append("CX",[stim.target_rec(meas_ind0-max_meas_ind),qubits[0],stim.target_rec(meas_ind1-max_meas_ind),qubits[1]])
                    self.stim_circuit.append("H",qubits)
                    self.stim_circuit.append("DEPOLARIZE1",qubits,self.gate_error)
                    self.stim_circuit.append("CX",[qubits[0],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[qubits[1],ancilla])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                    self.stim_circuit.append("M",ancilla)
                    self.stim_circuit.append("CX",[stim.target_rec(-1),qubits[0]])
                    max_meas_ind+=1
                elif basis=="Z":
                    if round>=0:
                        # qubits are reset
                        meas_ind0,meas_ind1 = self.measure_dict[(link_dict['anc_coord'],round)]
                        self.stim_circuit.append("CX",[stim.target_rec(meas_ind0-max_meas_ind),qubits[0],stim.target_rec(meas_ind1-max_meas_ind),qubits[1]])
                    self.stim_circuit.append("CX",[ancilla,qubits[0]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[0]],self.gate_error)
                    self.stim_circuit.append("CX",[ancilla,qubits[1]])
                    self.stim_circuit.append("DEPOLARIZE2",[ancilla, qubits[1]],self.gate_error)
                    self.stim_circuit.append("H",ancilla)
                    self.stim_circuit.append("DEPOLARIZE1",ancilla,self.gate_error)
                    self.stim_circuit.append("X_ERROR",ancilla,self.RO_error)
                    self.stim_circuit.append("M",ancilla)
                    self.stim_circuit.append("CZ",[stim.target_rec(-1),qubits[0]])
                    max_meas_ind+=1
            else:
                if round>=0:
                    # qubits are reset
                    meas_ind0 = self.measure_dict[(link_dict['anc_coord'],round)][0]
                    self.stim_circuit.append("CX",[stim.target_rec(meas_ind0-max_meas_ind),qubits[0]])
                if basis=="X":
                    self.stim_circuit.append("H",qubits[0])
                    self.stim_circuit.append("DEPOLARIZE1",qubits[0],self.gate_error)
        self.stim_circuit.append("TICK")

    def add_dets_obs(self,hex_fin_list,hex_start_list,basis,obs_links,ancillas,round):
        max_round_meas_ind = self.stim_circuit.num_measurements
        for hex, hex_start in zip(hex_fin_list, hex_start_list):
            if all([anc in ancillas for anc in hex]):
                if self.log_obs == basis and round<4.5:
                    det_targets = [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round)]]]
                    if det_targets:
                        self.stim_circuit.append("DETECTOR", det_targets)
                elif round>=4.5:
                    det_targets = ([stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex for target in self.measure_dict[(anc,round)]]]+
                                   [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in hex_start for target in self.measure_dict[(anc,round-4)]]])
                    if det_targets:
                        self.stim_circuit.append("DETECTOR", det_targets)
        if self.log_obs == basis and round>=1 and round%1!=0.5: #having this update in round-0 reduces the code distance...
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in obs_links for target in self.measure_dict[(anc,round)]]],0)

        if self.log_obs == basis and round>=.5 and round%1!=0.: #having this update in round-0 reduces the code distance...
            self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_target-max_round_meas_ind) for rec_target in [target for anc in obs_links for target in self.measure_dict[(anc,round)]]],1)
