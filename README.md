# QueueingCPN
A framework for discovering queuing-enabled coloured stochastic Petri nets from scheduled event logs. Scheduled event logs contain besides the real execution times, also the planned/scheduled ones.

## Dependencies
The framework requires that the PMlab toolkit is available in your python repository. 
It can be obtained here: http://www.cs.upc.edu/~jcarmona/PMLAB/
PMlab is GPL v3 licensed. 

The second dependency of QueueingCPN is to the SNAKES toolkit https://github.com/fpom/snakes
You need to install SNAKES and add the plugin "time_cpn.py" to the plugins directory of SNAKES.

## License
(c) 2015 Andreas Rogge-Solti (L-GPL v2.1)

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
