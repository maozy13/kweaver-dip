import { EllipsisOutlined } from '@ant-design/icons'
import { Popover, Tooltip } from 'antd'
import clsx from 'clsx'
import { useMemo } from 'react'
import intl from 'react-intl-universal'
import { type NavigateFunction, useLocation, useNavigate } from 'react-router-dom'

import type { SidebarPinnedDigitalHuman } from '@/apis/dip-studio/user'
import AppIcon from '@/components/AppIcon'
import IconFont from '@/components/IconFont'
import {
  SIDEBAR_OPEN_DH_SESSION_LOCATION_KEY,
  SIDEBAR_REOPEN_DH_SESSION_LOCATION_KEY,
} from '@/routes/types'
import { usePinnedDigitalHumansStore } from '@/stores/pinnedDigitalHumansStore'
import { resolveDigitalHumanIconSrc } from '@/utils/digital-human/resolveDigitalHumanIcon'
import { formatTotalDisplay } from '../utils'

/** 侧栏钉选区默认展开条数，超出部分收纳到「更多」，悬浮查看全部 */
const SIDEBAR_PINNED_VISIBLE_MAX = 3

export interface PinnedDigitalHumansSectionProps {
  /** 钉选行数据（服务端已组合档案；为空时父级不渲染本区块） */
  items: SidebarPinnedDigitalHuman[]
}

function PinnedDigitalHumanRow({
  human,
  highlightEmployeeId,
  navigate,
  onUnpin,
}: {
  human: SidebarPinnedDigitalHuman
  highlightEmployeeId: string
  navigate: NavigateFunction
  onUnpin: (id: string) => void
}) {
  const avatarSrc = resolveDigitalHumanIconSrc(human.icon_id)
  const isActive =
    highlightEmployeeId !== '' && highlightEmployeeId === human.id
  const unpinLabel = intl.get('sider.unpin')

  return (
    <div
      className={clsx(
        'group flex h-[36px] w-full items-center gap-0.5 rounded-lg px-2',
        isActive
          ? 'bg-[#f1f7fe] text-[--dip-primary-color]'
          : 'bg-transparent hover:bg-[#F5F5F5]',
      )}
    >
      <button
        type="button"
        onClick={() => {
          const path = `/studio/digital-human/${human.id}`
          if (highlightEmployeeId !== '' && highlightEmployeeId === human.id) {
            navigate(path, {
              replace: true,
              state: {
                [SIDEBAR_OPEN_DH_SESSION_LOCATION_KEY]: true,
                [SIDEBAR_REOPEN_DH_SESSION_LOCATION_KEY]: Date.now(),
              },
            })
            return
          }
          navigate(path, {
            state: {
              [SIDEBAR_OPEN_DH_SESSION_LOCATION_KEY]: true,
            },
          })
        }}
        className={clsx(
          'flex flex-1 min-w-0 items-center gap-2 border-0 bg-transparent cursor-pointer text-left rounded-lg p-0',
          isActive ? 'text-[--dip-primary-color]' : 'text-[rgba(0,0,0,0.88)]',
        )}
      >
        <div className="h-5 w-5 flex-shrink-0 overflow-hidden rounded-md bg-[rgba(0,0,0,0.06)]">
          {avatarSrc ? (
            <img src={avatarSrc} alt="" className="h-5 w-5 object-cover" />
          ) : (
            <AppIcon
              name={human.name}
              size={20}
              shape="square"
              className={clsx(
                'h-5 w-5 shrink-0',
                /* design token 前缀为 dip-avatar，内联 18px 需 !important 覆盖为 14px */
                '[&_.dip-avatar]:!text-[14px]',
                '[&_.dip-avatar]:leading-none',
              )}
            />
          )}
        </div>
        <span className="flex-1 min-w-0 truncate text-sm" title={human.name}>
          {human.name}
        </span>
      </button>
      <Tooltip title={unpinLabel} placement="bottom">
        <button
          type="button"
          className={clsx(
            'w-6 h-6 shrink-0 inline-flex items-center justify-center rounded border-0 bg-transparent cursor-pointer transition-all',
            'text-[rgba(0,0,0,0.45)] hover:bg-[rgba(0,0,0,0.04)]',
            'opacity-0 group-hover:opacity-100 focus-visible:opacity-100',
          )}
          aria-label={unpinLabel}
          onClick={(e) => {
            e.stopPropagation()
            void onUnpin(human.id)
          }}
        >
          <IconFont type="icon-solid-pin" className="text-sm" aria-hidden />
        </button>
      </Tooltip>
    </div>
  )
}

/** Studio 侧栏钉选数字员工列表（无独立区块标题，与菜单「数字员工」避免重复文案）。 */
export const PinnedDigitalHumansSection = ({ items }: PinnedDigitalHumansSectionProps) => {
  const navigate = useNavigate()
  const location = useLocation()
  const unpinSidebarDigitalHuman = usePinnedDigitalHumansStore((s) => s.unpinSidebarDigitalHuman)

  /** 详情页会话路由：`/studio/digital-human/:id`（与 `/studio/conversation?employee=` 区分开） */
  const highlightEmployeeId = useMemo(() => {
    const match = /^\/studio\/digital-human\/([^/]+)$/.exec(location.pathname)
    if (!match) return ''
    const id = match[1]?.trim() || ''
    if (!id || id === 'setting') return ''
    return id
  }, [location.pathname])

  const visibleItems = useMemo(
    () => items.slice(0, SIDEBAR_PINNED_VISIBLE_MAX),
    [items],
  )
  const overflowItems = useMemo(() => items.slice(SIDEBAR_PINNED_VISIBLE_MAX), [items])

  if (items.length === 0) {
    return null
  }

  const morePopoverContent =
    overflowItems.length === 0 ? null : (
      <div
        className="flex flex-col gap-0.5 py-0 min-w-[200px] max-w-[260px] max-h-[min(60vh,320px)] overflow-y-auto dip-hideScrollbar"
        aria-label={intl.get('sider.pinnedDigitalHumans.morePopoverAria')}
      >
        {overflowItems.map((human) => (
          <PinnedDigitalHumanRow
            key={`pinned-digital-human-overflow-${human.id}`}
            human={human}
            highlightEmployeeId={highlightEmployeeId}
            navigate={navigate}
            onUnpin={(id) => {
              void unpinSidebarDigitalHuman(id)
            }}
          />
        ))}
      </div>
    )

  return (
    <div className="px-2 py-1" aria-label={intl.get('sider.pinnedDigitalHumans.sectionAria')}>
      <div className="flex flex-col gap-1">
        {visibleItems.map((human) => (
          <PinnedDigitalHumanRow
            key={`pinned-digital-human-${human.id}`}
            human={human}
            highlightEmployeeId={highlightEmployeeId}
            navigate={navigate}
            onUnpin={(id) => {
              void unpinSidebarDigitalHuman(id)
            }}
          />
        ))}
        {overflowItems.length > 0 ? (
          <Popover
            trigger={['hover', 'click']}
            placement="rightTop"
            mouseEnterDelay={0.12}
            mouseLeaveDelay={0.25}
            overlayClassName="pinned-digital-humans-more-popover"
            overlayInnerStyle={{ padding: '6px 8px' }}
            arrow={false}
            content={morePopoverContent}
          >
            <button
              type="button"
              className={clsx(
                'flex h-[36px] w-full items-center gap-2 rounded-lg px-2 border-0 bg-transparent cursor-pointer text-left',
                'text-[rgba(0,0,0,0.88)] hover:bg-[#F5F5F5]',
              )}
              aria-label={intl.get('sider.pinnedDigitalHumans.moreRowAria', {
                count: formatTotalDisplay(overflowItems.length),
              })}
            >
              <span className="inline-flex shrink-0 items-center justify-center">
                <EllipsisOutlined className="text-lg text-[rgba(0,0,0,0.88)]" aria-hidden />
              </span>
              <span className="flex-1 min-w-0 truncate text-sm">
                {intl.get('sider.workPlan.more')}
              </span>
            </button>
          </Popover>
        ) : null}
      </div>
    </div>
  )
}
