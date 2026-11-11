#!/usr/bin/env python3
"""Generate a dependency-free native Xcode project from the checked-in Swift sources."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parents[1]
project = root / 'NemotronParseStudio.xcodeproj'
project.mkdir(exist_ok=True)
def uid(s): return hashlib.sha1(s.encode()).hexdigest()[:24].upper()
def q(s): return json.dumps(str(s))
objects=[]
def obj(name,value): objects.append(f'{uid(name)} = {{ {value} }};'); return uid(name)
files=sorted(root.glob('Sources/ParseCore/*.swift'))+sorted(root.glob('Sources/ParseStudio/*.swift'))
refs=[]; builds=[]
for p in files:
    rel=str(p.relative_to(root)); ref=obj(rel,f'isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = {q(rel)}; sourceTree = SOURCE_ROOT;')
    refs.append(ref); builds.append(obj('build'+rel,f'isa = PBXBuildFile; fileRef = {ref};'))
product=obj('product','isa = PBXFileReference; explicitFileType = wrapper.application; path = "Nemotron Parse Studio.app"; sourceTree = BUILT_PRODUCTS_DIR;')
products=obj('products',f'isa = PBXGroup; children = ({product},); name = Products; sourceTree = "<group>";')
group=obj('group',f'isa = PBXGroup; children = ({",".join(refs)},{products},); sourceTree = "<group>";')
sources=obj('sources',f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(builds)},); runOnlyForDeploymentPostprocessing = 0;')
frameworks=obj('frameworks','isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0;')
resources=obj('resources','isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0;')
projectConfigs=[]; targetConfigs=[]
for mode in ['Debug','Release']:
    projectConfigs.append(obj('project'+mode, f'isa = XCBuildConfiguration; name = {mode}; buildSettings = {{ SDKROOT = macosx; MACOSX_DEPLOYMENT_TARGET = 14.0; SWIFT_VERSION = 5.0; CLANG_ENABLE_MODULES = YES; }};'))
    targetConfigs.append(obj('target'+mode, f'''isa = XCBuildConfiguration; name = {mode}; buildSettings = {{ PRODUCT_NAME = "Nemotron Parse Studio"; PRODUCT_BUNDLE_IDENTIFIER = com.local.NemotronParseStudio; GENERATE_INFOPLIST_FILE = NO; INFOPLIST_FILE = Resources/Info.plist; CODE_SIGN_STYLE = Automatic; CODE_SIGN_IDENTITY = "-"; ENABLE_APP_SANDBOX = NO; SWIFT_EMIT_LOC_STRINGS = YES; SWIFT_OPTIMIZATION_LEVEL = {q('-Onone' if mode=='Debug' else '-O')}; COMBINE_HIDPI_IMAGES = YES; LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/../Frameworks"; }};'''))
pc=obj('projectconfigs',f'isa = XCConfigurationList; buildConfigurations = ({",".join(projectConfigs)},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
tc=obj('targetconfigs',f'isa = XCConfigurationList; buildConfigurations = ({",".join(targetConfigs)},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target=obj('target',f'isa = PBXNativeTarget; buildConfigurationList = {tc}; buildPhases = ({sources},{frameworks},{resources},); buildRules = (); dependencies = (); name = NemotronParseStudio; productName = NemotronParseStudio; productReference = {product}; productType = "com.apple.product-type.application";')
pr=obj('project',f'isa = PBXProject; attributes = {{ BuildIndependentTargetsInParallel = YES; LastSwiftUpdateCheck = 2700; LastUpgradeCheck = 2700; }}; buildConfigurationList = {pc}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; knownRegions = (en,Base,); mainGroup = {group}; productRefGroup = {products}; projectDirPath = ""; projectRoot = ""; targets = ({target},);')
(project/'project.pbxproj').write_text('// !$*UTF8*$!\n{ archiveVersion = 1; classes = {}; objectVersion = 56; objects = {\n'+'\n'.join(objects)+f'\n}}; rootObject = {pr}; }}\n')
import plistlib
(root/'Resources').mkdir(exist_ok=True)
with (root/'Resources/Info.plist').open('wb') as out:
    plistlib.dump({'CFBundleName':'Nemotron Parse Studio','CFBundleDisplayName':'Nemotron Parse Studio','CFBundleIdentifier':'$(PRODUCT_BUNDLE_IDENTIFIER)','CFBundleExecutable':'$(EXECUTABLE_NAME)','CFBundlePackageType':'APPL','CFBundleShortVersionString':'1.0.0','CFBundleVersion':'1','LSMinimumSystemVersion':'14.0','NSHighResolutionCapable':True,'NSPrincipalClass':'NSApplication','StudioSourceRoot':'$(SRCROOT)','NSAppTransportSecurity':{'NSAllowsLocalNetworking':True},'CFBundleDocumentTypes':[{'CFBundleTypeName':'Documents','CFBundleTypeRole':'Viewer','LSHandlerRank':'Alternate','LSItemContentTypes':['com.adobe.pdf','public.image','org.openxmlformats.wordprocessingml.document','org.openxmlformats.presentationml.presentation','org.openxmlformats.spreadsheetml.sheet']}]},out)
schemes=project/'xcshareddata/xcschemes'; schemes.mkdir(parents=True,exist_ok=True)
(schemes/'NemotronParseStudio.xcscheme').write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2700" version="1.3"><BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="Nemotron Parse Studio.app" BlueprintName="NemotronParseStudio" ReferencedContainer="container:NemotronParseStudio.xcodeproj"/></BuildActionEntry></BuildActionEntries></BuildAction><LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="Nemotron Parse Studio.app" BlueprintName="NemotronParseStudio" ReferencedContainer="container:NemotronParseStudio.xcodeproj"/></BuildableProductRunnable></LaunchAction><AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/></Scheme>''')
print(project)
