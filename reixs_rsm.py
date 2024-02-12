import numpy as np
import xrayutilities as xu
from silx.io.specfile import SpecFile,Scan
import pandas as pd
import os
import plotly
import plotly.graph_objects as go

import glob
from PIL import Image

MCP_coeff = np.loadtxt('flatfield.csv', delimiter = ',')

def unpack_spec_file(file_name):
    # Unpack spec file with name file_name+'.dat' into csv files with motors and counter values corresponding to different scans. 
    # Also, unpack mcp files with name file_name+'.dat_mcp' into different scans as well.
    # The destination is 'python_code/'+file_name+'_scans'
    file = open(file_name+'.dat','rb')
    
    try:
        os.mkdir('python_code/'+file_name+'_scans')
    except:
        print('Folder existed. Continue exporting...')

    text=[]

    for line in file:
        if len(line)>1:
            txtline = str(line,'utf-8')
            if txtline[1]=='S':
                info = str.split(txtline)
                scan_num=int(info[1])
                text=[]
    #             print(scan_num)
            text.append(txtline)

            if txtline[1]=='L':
                keys=str.split(txtline[3:-1])
                dataline=str(file.readline(),'utf-8')
                text.append(dataline)
                if ((dataline[0]!='#') and (dataline!='\n')):
                    data=np.fromstring(dataline, dtype=float, sep=' ')
                    num_column=len(data)
                    dataline=str(file.readline(),'utf-8')
                    text.append(dataline)
                    while (len(dataline)>1)&(dataline[0]!='#'):
                        new_data=np.fromstring(dataline, dtype=float, sep=' ')
                        data=np.append(data,new_data)
                        dataline=str(file.readline(),'utf-8')
                        text.append(dataline)
                        if (len(dataline)==0):
                            break
                    data=np.reshape(data,(-1,num_column))
                    scans=pd.DataFrame(data,columns=keys,index=None)
                    if os.path.isfile('python_code/'+file_name+'_scans/scan_'+str(scan_num)+'.csv')== False:
                        scans.to_csv('python_code/'+file_name+'_scans/scan_'+str(scan_num)+'.csv',index=False)

                        textfile = open('python_code/'+file_name+'_scans/scan_'+str(scan_num)+'_spec.txt','w')
                        textfile.writelines(text)
                        textfile.close()
    file.close()
    
    file = open(file_name+'.dat_mcp','rb')
    for line in file:
        if len(line)>1:
            txtline = str(line,'utf-8')
            if txtline[1]=='S':
                info = str.split(txtline)
                scan_num=int(info[1])
                if (scan_num>1) and len(text)>127:
#                     print(scan_num-1, len(text))
                    textfile = open('python_code/'+file_name+'_scans/scan_'+str(scan_num-1)+'_mcp.txt','w')
                    textfile.writelines(text)
                    textfile.close()
                n_pts = int(info[-2])+1
                text=[]
            if txtline[:3]=='#@I':
                for i in np.arange(128):
                    textline=str(file.readline(),'utf-8')
                    text.append(textline)

    textfile = open('python_code/'+file_name+'_scans/scan_'+str(scan_num)+'_mcp.txt','w')
    textfile.writelines(text)
    textfile.close()
    file.close()
    print('Export finished.')
    return None


def load_convert(file_name, scan_num, flatfield = True):
    # it loads a certain scan with MCP images and calculate the corresponding h,k,l coordinates
    
#     =============== load images =============
    imgs = np.genfromtxt('python_code/'+file_name+'_scans/scan_'+str(scan_num)+'_mcp.txt', delimiter = ' ')
    imgs = imgs.reshape([-1,128,128])
    if flatfield:
        for idx in np.arange(len(imgs)):
            imgs[idx] *= MCP_coeff
    # ============ load spec file and motor position====================
    sf = SpecFile(file_name + '.dat')
    scan = sf[scan_num-1]

    if np.shape(scan.data)[1] != np.shape(imgs)[0]:
        print('Number of images does not equal scan point number.')
        return None
    
    for line in scan.header:
        if line[:3] == '#G3':
            UB = np.array(line.split(' ')[-9:]).astype(float).reshape([3,3])
        if line[:3] == '#P0':
            angles = np.array(line.split(' ')[1:-2]).astype(float)

    energy = np.nanmean(scan.data_column_by_name('BeamEngy'))

    try:
        tt = scan.data_column_by_name('TwoTheta')
    except:
        tt = angles[0] * np.ones(np.shape(scan.data)[1])

    try:
        eta = scan.data_column_by_name('Theta')
    except:
        eta = angles[1] * np.ones(np.shape(scan.data)[1])

    try:
        chi = scan.data_column_by_name('Chi')
    except:
        chi = angles[2] * np.ones(np.shape(scan.data)[1])

    try:
        phi = scan.data_column_by_name('Phi')
    except:
        phi = angles[3] * np.ones(np.shape(scan.data)[1])
        
        
#     ================= load diffractometer geometry ==================
        
    qconversion = xu.QConversion(sampleAxis = ['z-','y+','z-'], detectorAxis = ['z-'], r_i = [0,1,0])

    hxrd = xu.HXRD( [0,1,0], [0,0,1], en = energy, qconv =  qconversion)

    hxrd.Ang2Q.init_area(
            'x+', 'z+',
            cch1=64, cch2=64,
            Nch1=128, Nch2=128,
            pwidth1=0.19685, pwidth2=0.19685,
            distance=300
        )
    
#     ================= angle to hkl ====================
    angle_values =   [eta, chi, phi, tt]   #[[26.056],  [13.028]]
    qx, qy, qz = hxrd.Ang2Q.area(*angle_values, UB=UB)
    return imgs, qx, qy, qz

def rsm_convert(file_name, scan_list, h_n = 50, k_n = 50, l_n = 50, 
            flatfield = True, return_imgs = False):

    # This program calculates the intensity at a gridded point with h_n*k_n*l_n.
    # The return is a 3d matrix, and 3* 1d lists of h,k,l.
    # input:
    # file_name: the spec file name
    # scan_list: can be a single scan number (integer) or list of number i.e. [14, 15, 16...]
    # h_n, k_n, l_n: the number of voxels in the output
    # flatfield: boolean, whether perform flatfield correction
    # return_imgs: boolean, whether return detector image in order to check the calculation
    if isinstance(scan_list, int):
        imgs, qx, qy, qz = load_convert(file_name, scan_list , flatfield)
    else:
        scan = scan_list[0]
        imgs, qx, qy, qz = load_convert(file_name, scan , flatfield)
        for scan in scan_list[1:]:
            imgs_temp, qx_temp, qy_temp, qz_temp = load_convert(file_name, scan , flatfield)
            imgs = np.vstack([imgs, imgs_temp])
            qx = np.vstack([qx, qx_temp])
            qy = np.vstack([qy, qy_temp])
            qz = np.vstack([qz, qz_temp])
    
#   ================= binning into regular grid ====================
    h_min,h_max = [np.min(qx), np.max(qx)]
    k_min,k_max = [np.min(qy), np.max(qy)]
    l_min,l_max = [np.min(qz), np.max(qz)]

    gridder = xu.Gridder3D(nx=h_n, ny=k_n, nz=l_n)
    gridder.KeepData(True)
    gridder.dataRange(
        xmin=h_min, xmax=h_max,
        ymin=k_min, ymax=k_max,
        zmin=l_min, zmax=l_max,
        fixed=True
    )
    flag = imgs>0
    gridder(qx[flag], qy[flag], qz[flag], imgs[flag])

    grid_data = gridder.data
    grid_data[grid_data<0.01]= np.nan
    coords = [gridder.xaxis, gridder.yaxis, gridder.zaxis]
    if return_imgs:
        return grid_data, coords, imgs, qx, qy, qz
    else:
        return grid_data, coords

def visualize_det(imgs, qx, qy, qz, cscale = [50, 99]):
    # This program views the loaded MCP image stack at corresponding hkl position.
    # The slider select the image frame.
    h_min,h_max = np.nanmin(qx), np.nanmax(qx)
    k_min,k_max = np.nanmin(qy), np.nanmax(qy)
    l_min,l_max = np.nanmin(qz), np.nanmax(qz)
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min
    cmin, cmax = np.nanpercentile(imgs, cscale)
    
    cmap = 'viridis'
    cmin = 0


    nb_frames = np.shape(imgs)[0]

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        x = qx[idx], 
        y = qy[idx], 
        z = qz[idx], 
        cmin=cmin, cmax=cmax, 
        surfacecolor=imgs[idx]
        ),
        name=str(idx) # you need to name the frame for the animation to behave properly
        )
        for idx in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        x = qx[0], 
        y =qy[0], 
        z=qz[0], 
        cmin=cmin, cmax=cmax, 
        surfacecolor=imgs[0]
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             width=600,
             height=600,
             scene=dict(
                        xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                        yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                        zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        xaxis_title='H',
                        yaxis_title='K',
                        zaxis_title='L'
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    # fig.show()
    return fig

def l_slice(grid_data, coords, logscale = False, dichro = False, title = None, start = 0, cscale = [50, 99]):
    # With the exported intensity grid points and h,k,l list, show l_slices.
    # logscale: show in log color scale.
    # dichro: whether this is a dichroic signal, if yes, the color scale is from (-cmax, +cmax)
    # title: string for figure title
    # start: int, the starting frame number
    # cscale: defalt [50, 99] set color scale corresponding to 50% and 99% intensity level.
    if logscale:
        volume = np.log(grid_data)
    else:
        volume = grid_data
    r, c = len(coords[0]), len(coords[1])
    
    h_min,h_max = [coords[0][0], coords[0][-1]]
    k_min,k_max = [coords[1][0], coords[1][-1]]
    l_min,l_max = [coords[2][0], coords[2][-1]]
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min
    
    if dichro:
        cmap = 'RdBu'
        cmin, cmax = np.nanpercentile(abs(volume), cscale)
        cmin = -cmax
    else:
        cmin, cmax = np.nanpercentile(volume, cscale)
        cmap = 'viridis'
        cmin = 0


    nb_frames = len(coords[2])
    xx,zz = np.meshgrid(coords[0],coords[1])
    # print(np.shape(xx), np.shape(zz),np.shape(volume[0,1,:]))

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(l_min + k/len(coords[2]) * llen) * np.ones((r, c)),
        surfacecolor=(volume[:,:,k]),
        cmin=cmin, cmax=cmax, 
        x = xx.T,#coords[2], 
        y= zz.T#coords[0]
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(l_min + start/len(coords[2]) * llen) * np.ones((r, c)),
        surfacecolor=(volume[:,:,start]),
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        colorbar=dict(thickness=20, ticklen=4), 
        x = xx.T,#coords[2], 
        y = zz.T #coords[0]
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title=title,
             width=600,
             height=600,
             scene=dict(
                        xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                        yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                        zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        xaxis_title='H',
                        yaxis_title='K',
                        zaxis_title='L'
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()
    return fig


def k_slice(grid_data, coords, logscale = False, dichro = False, title = None, start = 0, cscale = [50, 99]):
    # With the exported intensity grid points and h,k,l list, show k_slices.
    # logscale: show in log color scale.
    # dichro: whether this is a dichroic signal, if yes, the color scale is from (-cmax, +cmax)
    # title: string for figure title
    # start: int, the starting frame number
    # cscale: defalt [50, 99] set color scale corresponding to 50% and 99% intensity level.
    if logscale:
        volume = np.log(grid_data)
    else:
        volume = grid_data
    r, c = len(coords[0]), len(coords[2])
    
    h_min,h_max = [coords[0][0], coords[0][-1]]
    k_min,k_max = [coords[1][0], coords[1][-1]]
    l_min,l_max = [coords[2][0], coords[2][-1]]
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min

    if dichro:
        cmap = 'RdBu'
        cmin, cmax = np.nanpercentile(abs(volume), cscale)
        cmin = -cmax
    else:
        cmin, cmax = np.nanpercentile(volume, cscale)
        cmap = 'viridis'
        cmin = 0


    nb_frames = len(coords[1])
    xx,zz = np.meshgrid(coords[0],coords[2])
    # print(np.shape(xx), np.shape(zz),np.shape(volume[:,0,:]))

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        y=(k_min + k/len(coords[1]) * klen) * np.ones((r, c)),
        surfacecolor=(volume[:,k,:]),
        cmin=cmin, cmax=cmax, 
        x = xx.T,#coords[2], 
        z= zz.T#coords[0]
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        y=(k_min + start/len(coords[1]) * klen) * np.ones((r, c)),
        surfacecolor=(volume[:,start,:]),
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        colorbar=dict(thickness=20, ticklen=4), 
        x = xx.T,#coords[2], 
        z= zz.T #coords[0]
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title=title,
             width=600,
             height=600,
             scene=dict(
                        xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                        yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                        zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        xaxis_title='H',
                        yaxis_title='K',
                        zaxis_title='L'
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()
    return fig

def h_slice(grid_data, coords, logscale = False, dichro = False, title = None, start = 0, cscale = [50, 99]):
    # With the exported intensity grid points and h,k,l list, show h_slices.
    # logscale: show in log color scale.
    # dichro: whether this is a dichroic signal, if yes, the color scale is from (-cmax, +cmax)
    # title: string for figure title
    # start: int, the starting frame number
    # cscale: defalt [50, 99] set color scale corresponding to 50% and 99% intensity level.
    if logscale:
        volume = np.log(grid_data)
    else:
        volume = grid_data
    r, c = len(coords[1]), len(coords[2])
    
    h_min,h_max = [coords[0][0], coords[0][-1]]
    k_min,k_max = [coords[1][0], coords[1][-1]]
    l_min,l_max = [coords[2][0], coords[2][-1]]
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min
    
    if dichro:
        cmap = 'RdBu'
        cmin, cmax = np.nanpercentile(abs(volume), cscale)
        cmin = -cmax
    else:
        cmin, cmax = np.nanpercentile(volume, cscale)
        cmap = 'viridis'
        cmin = 0

    nb_frames = len(coords[0])
    xx,zz = np.meshgrid(coords[1],coords[2])
    # print(np.shape(xx), np.shape(zz),np.shape(volume[0,1,:]))

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        x=(h_min + k/len(coords[0]) * hlen) * np.ones((r, c)),
        surfacecolor=(volume[k,:,:]),
        cmin=cmin, cmax=cmax, 
        y = xx.T,#coords[2], 
        z= zz.T#coords[0]
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        x=(h_min + start/len(coords[0]) * hlen) * np.ones((r, c)),
        surfacecolor=(volume[start,:,:]),
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        colorbar=dict(thickness=20, ticklen=4), 
        y = xx.T,#coords[2], 
        z= zz.T #coords[0]
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title=title,
             width=600,
             height=600,
             scene=dict(
                        xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                        yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                        zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        xaxis_title='H',
                        yaxis_title='K',
                        zaxis_title='L'
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()
    return fig

def make_gif(frame_folder):
    # create gif from pictures in the folder
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}*.png")]
    frame_one = frames[0]
    frame_one.save(frame_folder+".gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    path = os.getcwd()
    print('Gif created in '+ path +'/'+ frame_folder+".gif")
    return None

def k_slice_gif(grid_data, coords, file_name, 
                logscale = False, dichro = False, cscale = [50, 99], start = 0, title = ''):
    # With the exported intensity grid points and h,k,l list, show l_slices.
    # logscale: show in log color scale.
    # dichro: whether this is a dichroic signal, if yes, the color scale is from (-cmax, +cmax)
    # title: string for figure title
    # start: int, the starting frame number
    # cscale: defalt [50, 99] set color scale corresponding to 50% and 99% intensity level.

    if logscale:
        volume = np.log(grid_data)
    else:
        volume = grid_data
    r, c = len(coords[0]), len(coords[2])
    
    h_min,h_max = [coords[0][0], coords[0][-1]]
    k_min,k_max = [coords[1][0], coords[1][-1]]
    l_min,l_max = [coords[2][0], coords[2][-1]]
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min
    
    
    if dichro:
        cmap = 'RdBu'
        cmin, cmax = np.nanpercentile(abs(volume), cscale)
        cmin = -cmax
    else:
        cmin, cmax = np.nanpercentile(volume, cscale)
        cmap = 'viridis'
        cmin = 0


    nb_frames = len(coords[1])
    xx,zz = np.meshgrid(coords[0],coords[2])
    
    def plot(k):
        fig = go.Figure()
        fig.add_trace(go.Surface(
            y=(k_min + k/len(coords[1]) * klen) * np.ones((r, c)),
            surfacecolor=(volume[:,k,:]),
            cmin=cmin, cmax=cmax, 
            x = xx.T,        z= zz.T
            )    )
        # Layout
        fig.update_layout(
                 title=title,
                 width=600,
                 height=600,
                 scene=dict(
                            xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                            yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                            zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            xaxis_title='H',
                            yaxis_title='K',
                            zaxis_title='L'
                            )
        )
        return fig

    for ii in range(len(coords[1])):
        frame = plot(ii)
        frame.write_image("image_export\\"+file_name+'_'+str(ii).zfill(2)+".png")

    make_gif('image_export/'+file_name)
    
    return None

def h_slice_gif(grid_data, coords, file_name, 
                logscale = False, dichro = False, cscale = [50, 99], start = 0, title = ''):

    if logscale:
        volume = np.log(grid_data)
    else:
        volume = grid_data
    r, c = len(coords[1]), len(coords[2])
    
    h_min,h_max = [coords[0][0], coords[0][-1]]
    k_min,k_max = [coords[1][0], coords[1][-1]]
    l_min,l_max = [coords[2][0], coords[2][-1]]
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min
    
    if dichro:
        cmap = 'RdBu'
        cmin, cmax = np.nanpercentile(abs(volume), cscale)
        cmin = -cmax
    else:
        cmin, cmax = np.nanpercentile(volume, cscale)
        cmap = 'viridis'
        cmin = 0

    nb_frames = len(coords[0])
    xx,zz = np.meshgrid(coords[1],coords[2])
    
    def plot(k):
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=(h_min + k/len(coords[0]) * hlen) * np.ones((r, c)),
            surfacecolor=(volume[k,:,:]),
            cmin=cmin, cmax=cmax, 
            y = xx.T,        z= zz.T
            )    )
        # Layout
        fig.update_layout(
                 title=title,
                 width=600,
                 height=600,
                 scene=dict(
                            xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                            yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                            zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            xaxis_title='H',
                            yaxis_title='K',
                            zaxis_title='L'
                            )
        )
        return fig

    for ii in range(len(coords[1])):
        frame = plot(ii)
        frame.write_image("image_export\\"+file_name+'_'+str(ii).zfill(2)+".png")

    make_gif('image_export/'+file_name)
    
    return None

def l_slice_gif(grid_data, coords, file_name, 
                logscale = False, dichro = False, cscale = [50, 99], start = 0, title = ''):

    if logscale:
        volume = np.log(grid_data)
    else:
        volume = grid_data
    r, c = len(coords[0]), len(coords[1])
    
    h_min,h_max = [coords[0][0], coords[0][-1]]
    k_min,k_max = [coords[1][0], coords[1][-1]]
    l_min,l_max = [coords[2][0], coords[2][-1]]
    hlen = h_max-h_min
    klen = k_max-k_min
    llen = l_max-l_min
    
    if dichro:
        cmap = 'RdBu'
        cmin, cmax = np.nanpercentile(abs(volume), cscale)
        cmin = -cmax
    else:
        cmin, cmax = np.nanpercentile(volume, cscale)
        cmap = 'viridis'
        cmin = 0

    nb_frames = len(coords[2])
    xx,zz = np.meshgrid(coords[0],coords[1])
    
    def plot(k):
        fig = go.Figure()
        fig.add_trace(go.Surface(
            z=(l_min + k/len(coords[2]) * llen) * np.ones((r, c)),
            surfacecolor=(volume[:,:,k]),
            cmin=cmin, cmax=cmax, 
            x = xx.T,        y= zz.T
            )    )
        # Layout
        fig.update_layout(
                 title=title,
                 width=600,
                 height=600,
                 scene=dict(
                            xaxis=dict(range=[h_min-abs(hlen)*0.05, h_max+abs(hlen)*0.05], autorange=False),
                            yaxis=dict(range=[k_min-abs(klen)*0.05, k_max+abs(klen)*0.05], autorange=False),
                            zaxis=dict(range=[l_min-abs(llen)*0.05, l_max+abs(llen)*0.05], autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            xaxis_title='H',
                            yaxis_title='K',
                            zaxis_title='L'
                            )
        )
        return fig

    for ii in range(len(coords[1])):
        frame = plot(ii)
        frame.write_image("image_export\\"+file_name+'_'+str(ii).zfill(2)+".png")

    make_gif('image_export/'+file_name)
    
    return None