#!/usr/bin/env python
import netCDF4 as nc, numpy as np
import pprint as pp
import sys


def compare_ncattr(varid1, varid2):
    atts1 = ({k: varid1.getncattr(k) for k in varid1.ncattrs()})
    atts2 = ({k: varid2.getncattr(k) for k in varid2.ncattrs()})

    messages = []
    # Attributes missing in one set, but present in the other
    attributes_missing = [(list(set(atts2)-set(atts1))), (list(set(atts1)-set(atts2)))]
    if attributes_missing[0]>[]:  
        messages.append('Missing attributes : ' + ','.join(attributes_missing[0]))
    if attributes_missing[1]>[]:  
        messages.append('Added attributes   : ' + ','.join(attributes_missing[1]))

    attributes_differ  = {}
    for attname in set(atts2).intersection(set(atts1)):
        type1 = type(atts1[attname])
        type2 = type(atts2[attname])
        if type1 != type2:
            messages.append('Attribute %s in reference has the wrong type' % attname)
            messages.append('    tested    : ' + str(type1))
            messages.append('    reference : ' + str(type2))
        differs=(atts1[attname] != atts2[attname]) 
        if hasattr(differs,'__iter__'):
            differs = any(differs)
        if differs: 
            attributes_differ[attname]=(atts1[attname], atts2[attname])
            messages.append('Attribute %s differs:'%attname)
            messages.append('    tested    : ' + str(atts1[attname]))
            messages.append('    reference : ' + str(atts2[attname]))
    return messages, attributes_differ, attributes_missing
        


def ncdiff(fn1,fn2,th,output,attrib):
   files_differ = False
   map1 = nc.Dataset(fn1)
   map2 = nc.Dataset(fn2)
   sys.stdout.write("%50s, %1s, %12s %12s %10s %20s %20s \n" % ('varname', ' ', 'maxdiff', 'mindiff', 'xvector', \
                     'value1', 'value2')) 
   for varname,varobj in map1.variables.items():
   #  if (varobj.datatype=='float64'):
      var1 = map1.variables[varname]
      try:
         if (not(varname in map2.variables.keys())):
            sys.stdout.write("%18s, %1s, %s\n" % (varname, ' ', ' No such variable in File 2 !!!!!'))
            files_differ = True
            continue
         else:
            var2 = map2.variables[varname]
         shape1 = np.shape(var1[:])
         shape2 = np.shape(var2[:])
         if shape1==shape2:
            shapes_differ = ""
            diff = var2[:]-var1[:]
         else:
            shapes_differ = "%1s, %s : %s = %s versus %s" % (' ','Shapes differ !!!!! ', "{0}".format(var1.dimensions),"{0}".format(shape1),"{0}".format(shape2))
            files_differ = True
            minshape = min(shape1,shape2)
            ndim=len(shape1)
            ndx = tuple([slice(0,minndx,None) for minndx in minshape])
            diff = var2[ndx]-var1[ndx]
         maxdiff = np.max(diff)
         mindiff = np.min(diff)
         if (len(np.shape(diff))>=1):
            if (max(abs(maxdiff),abs(mindiff)) > th):
               vector = np.unravel_index(np.argmax(abs(diff)),np.shape(diff))
               xvector = "{0}".format(vector)
               files_differ = True
               output.write("%50s, %1s, %12.5e %12.5e %10s %20.10e %20.10e %s\n" % (varname, 'X', maxdiff, mindiff, xvector, \
                                                                var1[vector], var2[vector], shapes_differ))
            else:
               output.write("%50s, %1s, %12.5e %12.5e %s\n" % (varname, '', maxdiff, mindiff, shapes_differ))
         if attrib:
             attmsg, attributes_differ, attributes_missing = compare_ncattr(var1, var2)
             if attmsg>[]:
                 for msg in attmsg:
                     output.write("    %s\n"%msg)
                 files_differ = True
         
      except:
         sys.stdout.write("%50s, %1s, %s\n" % (varname, ' ', 'No results !!!!!'))
    
   if attrib:
       attmsg, attributes_differ, attributes_missing = compare_ncattr(map1, map2)
       if attmsg>[]:
           output.write("Global attributes differ:\n")
           for msg in attmsg:
               output.write("    %s\n"%msg)
           files_differ = True
   map1.close()
   map2.close()
   return files_differ


attrib = False
for sysarg in sys.argv:
    if 'attrib=yes' in sysarg:
        attrib=True
if ncdiff(sys.argv[1],sys.argv[2],1.e-10,sys.stdout,attrib):
    print ("Files differ")
else:
    print ("Files are same")




      
       
 




